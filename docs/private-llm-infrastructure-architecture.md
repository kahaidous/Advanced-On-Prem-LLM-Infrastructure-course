# Dimensionner une infrastructure LLM on-premises

Résumé du cours « Advanced On-Premises LLM Infrastructure », séance 1.

## 1. Objectif

Savoir déduire le matériel (type et nombre de GPU, serveur, réseau, capacité de réserve) à partir de la **charge de travail**, et non l'inverse. Chaque choix doit pouvoir être défendu par un chiffre.

```
Charge de travail          Chiffres                   Matériel
req/s, tokens prompt  →   Go de mémoire          →   type et nombre de GPU
tokens sortie,            To/s de bande passante     serveur, réseau
cibles de latence         FLOP/s de calcul           topologie
                          secondes de chargement     capacité de réserve
```

**Exemple:** : un assistant RAG (Retrieval-Augmented Generation) pour une banque.

| Donnée | Valeur |
|---|---|
| Débit de pointe | 4 req/s |
| Prompt (p95) | 3 000 tokens |
| Sortie (p95) | 400 tokens |
| Objectifs de latence (SLO) | TTFT ≤ 2 s, TPOT ≤ 50 ms |
| Disponibilité | 99,9 % (heures ouvrées) |
| Modèle | 70B dense, GQA : 80 couches, 8 têtes KV, dimension de tête 128 |
| Précision | poids FP8 et KV cache FP8 |
| Contraintes | aucun accès internet, survivre à la perte d'un nœud |

**Résultat final** : 8 GPU H200 (2 nœuds de 4), 4 réplicas en TP=2. 

Un dimensionnement basé uniquement sur la mémoire aurait donné 2 GPU, soit 4 fois trop peu.

## 2. Les 7 étapes à suivre

| # | Étape | Question à résoudre | Résultat sur l'exemple |
|---|---|---|---|
| 1 | Charge de travail | Combien de requêtes en vol ? (In flight) | 117 séquences (conception) |
| 2 | Mémoire | Poids + KV cache + marge tiennent-ils ? | 2 × H200 suffisent côté mémoire |
| 3 | Calcul | Le prefill et le decode tiennent-ils les SLO ? | Prefill à 107 % sur 2 GPU, donc 4 GPU |
| 4 | Mesure | Les débits publiés confirment-ils ? | Oui : 4 GPU, pas 2 |
| 5 | Hôte | CPU, RAM, stockage, réseau, puissance par nœud | 24 cœurs, 1,1 To de RAM, NVMe local |
| 6 | Déploiement | Plans, topologie, configuration | Kubernetes en air-gap |
| 7 | Panne | Combien de réserve pour perdre un nœud ? | 8 GPU, 4 réplicas |

## 3. Détail de chaque étape

### Étape 1 : traduire la charge en requêtes en vol

Les requêtes simultanées déterminent la mémoire. On les calcule avec la **loi de Little** :

```
L = λ × W
W = TTFT + n_out × TPOT
```

- λ : taux d'arrivée (req/s), L : requêtes en vol, W : temps passé dans le système.
- On planifie avec des valeurs mesurées sous charge, pas avec les limites des SLO : TTFT = 0,6 s et TPOT = 35 ms (les SLO autorisent 2 s et 50 ms).
- Exemple : W = 0,6 + 400 × 0,035 = **14,6 s** ; L = 4 × 14,6 ≈ **58** en moyenne.
- On dimensionne sur **2L ≈ 117** pour absorber les rafales, car la loi de Little ne donne qu'une moyenne.
- Cas des agents : la concurrence se multiplie par le nombre d'appels parallèles (5 runs/s × 60 s × 4 appels = 1 200 requêtes en vol).
- On évalue sur le **goodput** (requêtes qui respectent tous les SLO), pas sur le débit brut (throughput) qui compte aussi les requêtes trop lentes.

### Étape 2 : vérifier que tout tient en mémoire GPU

Le bon test est : **poids + KV cache des requêtes en vol + marge ≤ mémoire HBM (High Bandwidth Memory)**. « Les poids tiennent » est un test insuffisant : il passe avec un utilisateur et échoue au pic.

**Poids** = paramètres × octets par paramètre.

| Format | Octets | 70B |
|---|---|---|
| BF16 | 2 | 141 Go |
| FP8 | 1 | 71 Go |
| 4 bits | 0,5 | ~35 Go (NVFP4 : ~40 Go) |

Pour un modèle MoE (Mixture of Experts), la mémoire suit les paramètres **totaux**, pas les paramètres actifs.

**KV cache par token** = 2 × n_couches × n_têtes_KV × d_tête × octets

- Exemple en FP8 : 2 × 80 × 8 × 128 × 1 = 163 840 octets = **160 Kio** par token (320 Kio en BF16).
- Le « 2 » correspond à K et V. L'oublier divise le résultat par deux.
- Les tokens du prompt comptent autant que ceux de la sortie : 3 000 + 400 = 3 400 tokens par séquence, soit 0,56 Go.
- Total : 0,557 Go × 117 ≈ **65 Go**.
- L'architecture d'attention change tout : GQA (Grouped-Query Attention) réduit les têtes KV ; MLA (Multi-head Latent Attention) réduit encore plus. À partir de la fiche du modèle, jamais du nombre de paramètres.
- Un KV cache en FP8 divise la mémoire par deux (vLLM : `--kv-cache-dtype fp8`).

**Budget sur 2 × H200 (282 Go)** :

```
282 × 0,9 (gpu_memory_utilization)   = 253,8 Go
− poids FP8                          =  71 Go
− activations et CUDA graphs         =   8 Go
= pool de KV cache                   ≈ 175 Go   (≥ 65 Go nécessaires, ~315 séquences possibles)
```

La mémoire ne limite donc pas ce cas. Si le pool se remplit, vLLM fait de la **préemption** (il évince une requête et recalcule son cache) : la latence de queue explose. Une préemption en hausse avec des requêtes en attente signifie que le dimensionnement est faux.

### Étape 3 : vérifier la vitesse (prefill et decode)

Les deux phases n'ont pas la même limite. On fait deux contrôles.

**Decode** (limité par la bande passante, fixe le TPOT) :

```
octets par pas = poids + KV vivant = 71 + 65 = 136 Go
bande passante effective = 2 × 4,8 To/s × 0,7 (MBU) = 6,7 To/s
temps par pas = 136 / 6 720 ≈ 20 ms  <  50 ms   → OK
```

**Prefill** (limité par le calcul, fixe le TTFT) :

```
demande = 4 req/s × 3 000 tokens × 2 × 70,6·10⁹ = 1,69 PFLOP/s
capacité 2 × H200 = 2 × 1 979 TFLOPS (FP8 dense) × 0,4 (MFU) = 1,58 PFLOP/s
utilisation = 107 %   → ÉCHEC (la file de prefill grossit sans limite)
4 × H200 : 3,17 PFLOP/s → 54 %   → OK
```

Points clés :
- Les coefficients de réduction (MFU = 40 %, MBU = 70 %) sont des **hypothèses de planification**, à remplacer par des mesures.
- Plafond mono-utilisateur du decode : bande passante / poids (4 800 / 71 ≈ 68 tokens/s sur un H200 ; mesuré : 51).
- Le **point de crête** (ridge) du modèle roofline = FLOP/s de pic / bande passante : en dessous, on est limité par la mémoire ; au-dessus, par le calcul.
- Les longs prompts pénalisent beaucoup plus que les longues sorties (4,8× contre 1,3× de débit en moins, sur H200).
- Si le prefill est la limite et pas le decode, les leviers sont, dans l'ordre : prefix caching, chunked prefill, plus de capacité de prefill, puis séparation prefill/decode (désagrégation).
- Passer d'un H200 à un B200 accélère le decode d'environ 1,6× (rapport des bandes passantes 7,7 / 4,8), pas du rapport des FLOPs. Le B300 ajoute de la mémoire, pas de bande passante.

### Étape 4 : confirmer par des mesures publiées

```
sortie nécessaire = 4 × 400 = 1 600 tokens/s
un H200 (70B FP8, prompts longs) ≈ 540 à 600 tokens/s
2 × H200 = 1 100 à 1 200 tokens/s   → ~30 % en dessous
4 × H200 = 2 200 à 2 400 tokens/s   → 1,4 × le besoin
```

- Deux méthodes (calcul et mesure) donnent la même réponse : 4 GPU pour le pic.
- Les chiffres MLPerf sont obtenus sur des séquences courtes avec des piles très optimisées : les diviser par 2 à 5 pour planifier.
- Avant l'achat, lancer un balayage de charge sur un GPU loué avec le vrai mélange de prompts (GuideLLM `--profile sweep`, `vllm bench serve`, ou inference-perf) et lire le point où le goodput cesse de monter.

### Étape 5 : dimensionner l'hôte

Par nœud (4 × H200) :

| Composant | Choix | Règle |
|---|---|---|
| GPU | 4 × H200, 2 paires TP=2 reliées par NVLink | TP jamais sur des chemins SYS |
| CPU | ≥ 24 cœurs physiques | 6 par GPU (guide NVIDIA-Certified) |
| RAM | ≥ 1,1 To | 2 × la mémoire GPU totale (4 × 141 = 564 Go) |
| Stockage | NVMe local en miroir, poids en cache | 71 Go ÷ 6,5 Go/s ≈ 11 s de transfert brut |
| Réseau | cartes séparées pour gestion et données | pas de RDMA (aucun réplica sur 2 nœuds) |

- **Topologie** : lancer `nvidia-smi topo -m` à la livraison. Le code SYS (traversée entre sockets CPU) est le plus lent ; un serveur 8 GPU en PCIe sans NVLink s'est comporté comme deux serveurs de 4 GPU.
- **Parallélisme** : le TP (Tensor Parallelism) reste dans le nœud sur NVLink ; seul le PP (Pipeline Parallelism) traverse le réseau.
- **Temps de chargement** (T_load) = transfert + compilation + capture des CUDA graphs + démarrage du moteur. Le chargeur compte plus que le disque : un modèle de 233 Gio a pris 226 s avec le chargeur par défaut et 37 s avec le Run:ai Model Streamer. À mesurer de bout en bout sur la vraie machine.
- **Puissance** : demander les kW par rack à l'hébergeur avant de choisir le GPU. Un DGX B200 consomme jusqu'à 14,3 kW, plus de deux fois ce qu'un ancien rack d'entreprise fournit (5 à 7 kW). Multiplier par le PUE (~1,5) pour la charge de l'installation.

### Étape 6 : déploiement, plans et topologie

**Trois plans séparés**, pour savoir quel plan est en panne :

| Plan | Rôle | Choix |
|---|---|---|
| Contrôle | décide quoi tourne où | Kubernetes ; etcd sur 3 nœuds CPU, hors des nœuds GPU |
| Inférence | transforme les requêtes en tokens | passerelle avec routage sensible au KV cache ; 4 réplicas TP=2 |
| Stockage | octets au repos | registre de modèles signé dans l'enclave ; cache NVMe par nœud |

Les requêtes ne traversent jamais le plan de contrôle ; les poids ne circulent qu'au chargement. Si le registre tombe, les réplicas en cours continuent, mais aucun nouveau ne peut démarrer. Dans les incidents analysés par Microsoft, environ 60 % des incidents graves venaient du moteur d'inférence : c'est lui qu'il faut surveiller en premier.

**Symptômes de plans mélangés** : HTTP 500 depuis un pod « Ready » (routage sur la santé du pod, pas sur la capacité KV), boucle de redémarrage au déploiement (sonde de vivacité qui tue un modèle en cours de chargement), pics de latence pendant les déploiements (poids tirés sur le réseau GPU), pas de replanification après une panne (etcd sur un nœud GPU).

**Sonde de démarrage** : le budget doit dépasser T_load.

```yaml
startupProbe:
  httpGet: {path: /health, port: 8000}
  periodSeconds: 10
  failureThreshold: 30     # 30 × 10 s = 300 s
```

**Topologies possibles** : bare metal (pilote, 1 à 2 serveurs), Kubernetes (nombreux réplicas, déploiements progressifs), cloud privé (VMware, OpenShift), air-gapped (défense, justice, banques centrales). Les régulations exigent résidence des données, audit et plan de sortie, rarement une topologie nommée (vérifier le texte du régulateur, car les sources du cours sont des blogs de fournisseurs). Notre banque : Kubernetes dans une enclave air-gapped.

**Air-gap** : tout vit dans l'enclave (registre de modèles avec import signé, miroirs de conteneurs et de paquets, modèle d'embedding local, base vectorielle, observabilité, PKI et heure internes). La fuite la plus fréquente est un appel à une API d'embedding distante. Variables à positionner, avec poids et tokenizer en chemin local :

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
vllm serve /models/llama-3.3-70b --tokenizer /models/llama-3.3-70b ...
```

Ces variables **réduisent** les appels sortants ; seul un blocage réseau suivi d'un test complet du flux de travail prouve zéro sortie.

**Configuration vLLM qui encode les étapes 1 à 3** :

```bash
vllm serve /models/llama-3.3-70b \
  --tensor-parallel-size 2 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --max-model-len 8192
```

`--max-num-seqs` est un plafond, pas un droit : 128 est juste au-dessus des 117 de conception, et le pool en soutient ~315. Promettre plus de séquences que le pool n'en soutient produit des préemptions.

### Étape 7 : survivre à une panne

**Domaines de panne**, du plus petit au plus grand : processus du moteur, GPU, nœud, rack ; et transversaux : réseau, stockage, plan de contrôle. Un réplica TP=8 tombe dès qu'un seul de ses 8 GPU tombe : garder le TP aussi étroit que possible.

**Règle de capacité** : pendant qu'un nœud est indisponible, les N − 1 réplicas restants doivent porter le pic, pendant toute la fenêtre de récupération (détection, planification, tirage de l'image, chargement, compilation, préchauffage).

```
pic = 2 réplicas (4 GPU)
un nœud perdu = 2 réplicas perdus
N = 2 + 2 = 4 réplicas = 2 nœuds × 4 H200 = 8 GPU
après panne d'un nœud : 2 réplicas portent le pic à ~54 % de charge de prefill
```

Faire survivre à un nœud a doublé la facture, car un domaine de panne contient la moitié de la flotte. Empaqueter densément les réplicas augmente le coût de la réserve : avec 3 réplicas nécessaires et 2 par nœud, il en faut 6.

**Taux de panne** (ordre de grandeur, entraînement Llama 3 : 419 interruptions en 54 jours sur 16 384 GPU, dont environ 59 % côté GPU) : ~4,7·10⁻⁴ par GPU et par jour, soit environ 2,8 GPU perdus par an sur 16 GPU, 11 sur 64, 1,4 sur nos 8. Rare un jour donné, certain sur une année : prévoir la réserve.

**Réception du matériel** : tester chaque serveur avant le trafic. Environ 7 % de 128 serveurs H100 neufs ont échoué aux tests d'un opérateur. Échelle : `dcgmi diag -r 1` (secondes), `-r 2` (~10 min), `-r 3` (~35 min), `-r 4` (~2 h), puis `gpu_burn`, tests NCCL (barre ~460 Go/s sur 8 × H100), puis votre benchmark vLLM.

**Surveillance** :
- `nvidia-smi` affiche « 100 % d'utilisation » dès qu'un seul noyau tourne (1 SM sur 132) : utiliser les champs DCGM (`PROF_SM_ACTIVE`, `PROF_PIPE_TENSOR_ACTIVE`, `PROF_DRAM_ACTIVE`).
- Côté vLLM : `num_requests_waiting`, `kv_cache_usage_perc`, `num_preemptions`, histogrammes de TTFT et TPOT.

**Codes XID (erreurs GPU)** :

| Classe | Codes | Action |
|---|---|---|
| Logiciel | 13, 31, 43 | redémarrer l'application |
| Matériel | 48, 64, 74, 79, 95, 119, 120 | vider le nœud, réinitialiser le GPU, RMA si répété |
| Information | 63, 92, 94 | consigner et suivre la tendance |

Exemple : XID 79 (le GPU a disparu du bus PCIe) → vider le nœud, redémarrer l'hôte, lancer les diagnostics DCGM.

**Exploitation dans la durée** : figer la branche du pilote (R580, support long terme jusqu'en juin 2028), charger uniquement du `safetensors` (les fichiers pickle peuvent exécuter du code), vérifier les signatures de modèles, corriger le moteur (CVE-2026-22778 sur vLLM, score 9,8), et placer un proxy d'authentification devant le moteur (`--api-key` ne protège que certaines routes).

## 4. Réponse finale et sensibilité

| Élément | Valeur |
|---|---|
| GPU | 8 × H200, 2 nœuds |
| Réplicas | 4, en TP=2, deux par nœud |
| Dimensionnement sur la mémoire seule | 2 GPU (4 fois trop peu) |
| Limite qui décide | calcul du prefill, à 54 % sur le nœud survivant |
| Chiffre le moins sûr | débit sur le vrai mélange de prompts |

**Si la charge change** :

| Changement | Nouvelle limite |
|---|---|
| Prompts de 32K tokens, 1 req/s | mémoire et prefill : ~411 Go de KV cache |
| Modèle MoE (~120B total, ~10B actifs) | mémoire pour tous les experts |
| Chat à 20 req/s, prompts courts | batch de decode proche du point de crête |
| Prompt système partagé de 1K tokens | prefill réduit d'un tiers avec le prefix caching |
| p95 des prompts de 3K à 8K tokens | prefill : 4,5 PFLOP/s demandés contre 3,17 disponibles (142 %), échec en premier |

## 5. À retenir

1. **Tenir** : poids + KV cache en vol + marge ≤ HBM.
2. **Aller assez vite** : le prefill (calcul) fixe le TTFT, le decode (bande passante) fixe le TPOT ; vérifier les deux.
3. **Survivre** : N − 1 réplicas portent le pic, temps de chargement inclus.
4. Partir de la charge, donner un chiffre à chaque décision, et confirmer par un benchmark avant l'achat.

## Abréviations

| Abréviation | Signification |
|---|---|
| API | Application Programming Interface |
| BF16 | bfloat16, format flottant sur 16 bits (2 octets) |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture (plateforme GPU de NVIDIA) |
| CVE | Common Vulnerabilities and Exposures |
| DCGM | Data Center GPU Manager (diagnostics et télémétrie NVIDIA) |
| DGX | appliance 8 GPU de NVIDIA |
| etcd | base clé-valeur répliquée qui stocke l'état de Kubernetes |
| FP8 | virgule flottante sur 8 bits (1 octet) |
| FLOP/s | opérations à virgule flottante par seconde |
| GB / Go | gigaoctet (10⁹ octets) |
| GPU | Graphics Processing Unit |
| GQA | Grouped-Query Attention |
| HBM | High Bandwidth Memory |
| HTTP | HyperText Transfer Protocol |
| KV cache | Key-Value cache (clés et valeurs d'attention des tokens en vol) |
| Kio | kibioctet (1 024 octets) |
| L | nombre de requêtes en vol (loi de Little) |
| λ | taux d'arrivée des requêtes (req/s) |
| LTS | Long-Term Support |
| MBU | Model Bandwidth Utilization (part de la bande passante de pic réellement atteinte) |
| MFU | Model FLOPs Utilization (part des FLOP/s de pic réellement atteinte) |
| MIG | Multi-Instance GPU |
| MLA | Multi-head Latent Attention |
| MoE | Mixture of Experts |
| NCCL | NVIDIA Collective Communications Library |
| NIC | Network Interface Card |
| NVMe | Non-Volatile Memory Express |
| NVFP4 | format 4 bits de NVIDIA (Blackwell) |
| PFLOP/s | 10¹⁵ opérations par seconde |
| PCIe | PCI Express |
| PKI | Public Key Infrastructure |
| PP | Pipeline Parallelism |
| PUE | Power Usage Effectiveness |
| RAG | Retrieval-Augmented Generation |
| RAM | Random Access Memory |
| RDMA | Remote Direct Memory Access |
| req/s | requêtes par seconde |
| RMA | Return Merchandise Authorization (retour du matériel défectueux) |
| SLO | Service Level Objective |
| SM | Streaming Multiprocessor |
| TP | Tensor Parallelism (ex. TP=2 : un réplica sur 2 GPU) |
| TPOT | Time Per Output Token |
| TTFT | Time To First Token |
| T_load | temps de chargement complet d'un réplica jusqu'au service |
| W | temps moyen passé dans le système par requête (loi de Little) |
| XID | code d'erreur GPU écrit par le pilote NVIDIA |