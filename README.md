# Hyper-Decept

HyperDecept: A cross-dimensional multimodal framework integrating LLM-native psychological profiling and hyperbolic graph learning for detecting coordinated multi-agent deception.

---
## What This Project Does

The core of HyperDecept lies in **three detection scripts** under `Character Classification/`:

| Script | Task | Method |
|--------|------|--------|
| `new_main_classifier.py` | **Binary classification** | XGBoost on 26-dim multi-modal features |
| `new_gang_detection.py` | **Bot gang detection** | Louvain community detection on bot subgraph |
| `new_role_assigner.py` | **Tactical role discovery** | HGT + Poincaré ball + DPMM clustering |

**Key innovations:**
- **4-dimensional psychological features** — Empathy Gap, Dark Triad, Emotional Contagion, Emotion Volatility (from `emotional_analysis/`) are fused into the feature matrix to enhance detection
- **Cosine-similarity augmented heterogeneous graph** — Feature vector cosine similarity is used to add edges between semantically similar nodes, compensating for sparse follow-graph connectivity

Everything else in the repository (profile generation, vector store, simulation engine, visualization) serves these three detection scripts by providing additional data sources or analytical outputs.

---
## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)
2. [Core: Multi-Modal Classification & Role Discovery](#core-multi-modal-classification--role-discovery-character-classification)
   - [Data Sources](#data-sources)
   - [Script 1: Binary Classifier](#script-1-binary-classifier-new_main_classifierpy)
   - [Script 2: Bot Gang Detection](#script-2-bot-gang-detection-new_gang_detectionpy)
   - [Script 3: Hyperbolic Role Discovery](#script-3-hyperbolic-role-discovery-new_role_assignerpy)
   - [Script 4: Ablation Study & Independence Verification](#script-4-ablation-study--independence-verification-ablation)
3. [Supplementary Modules](#supplementary-modules)
   - [Step A: Deep Persona Generation (Deeppersona)](#step-a-deep-persona-generation-deeppersona)
   - [Step B: Semantic Vector Store (deeppersona_ai)](#step-b-semantic-vector-store-deeppersona_ai)
   - [Step C: Multi-Agent Simulation (MultiAgent4Collusion)](#step-c-multi-agent-simulation-multiagent4collusion)
   - [Step D: Emotional/Psychological Analysis (emotional_analysis)](#step-d-emotionalpsychological-analysis-emotional_analysis)
   - [Step E: Data Adapter (data_processing)](#step-e-data-adapter-data_processing)

---
## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────── Core Pipeline ───────────┐                                    │
│  │                                     │                                    │
│  │  Character Classification/          │                                    │
│  │    new_main_classifier.py  (main)   │  ← XGBoost binary classification   │
│  │    new_gang_detection.py   (gang)   │  ← Louvain gang detection          │
│  │    new_role_assigner.py    (role)   │  ← HGT + Poincaré + DPMM          │
│  │                                     │                                    │
│  │  Input: DB + CSV (72-agent demo     │                                    │
│  │         or TwiBot-22 processed)     │                                    │
│  └─────────────────────────────────────┘                                    │
│              │                                                              │
│  ┌─────────── Supplementary Modules ───┐                                    │
│  │ ① Deeppersona           (profiles)  │  → agents.json                     │
│  │ ② deeppersona_ai        (vector DB) │  → vector_store/                   │
│  │ ③ MultiAgent4Collusion  (simulation)│  → .db + .csv (feeds core)         │
│  │ ④ emotional_analysis    (psychology)│  → feature vectors (feeds core)     │
│  │ ⑤ data_processing       (adapter)   │  → standardized DB + CSV           │
│  └─────────────────────────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---
## Prerequisites & Installation

### System Requirements

- **Python**: 3.10+ (recommended 3.11)
- **GPU**: CUDA-compatible GPU recommended (CPU fallback available)
- **RAM**: 16GB+ minimum, 32GB+ for large-scale simulation

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### HuggingFace Mirror (if blocked in your region)

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

---
## Core: Multi-Modal Classification & Role Discovery (`Character Classification`)

This is the **central module** of HyperDecept. It performs coordinated deception detection through three scripts:

1. **Binary classification** (main) — XGBoost on 26-dim multi-modal features
2. **Bot gang detection** (gang) — Louvain community detection on bot subgraph
3. **Hyperbolic role discovery** (role) — HGT + Poincaré ball + DPMM clustering

### Data Sources

The pipeline accepts two kinds of input:

#### Mode A: Demo Data

The `data/` directory (included in this repository) contains a pre-packaged 72-agent dataset:

| File | Description |
|------|-------------|
| `data/test_72.db` | SQLite database with user, post, follow, like tables |
| `data/72agent_deeppersonal.csv` | Multi-modal feature CSV with `user_id`, `user_char`, `previous_tweets`, `user_type` (good/bad) |

Use this to quickly verify the pipeline. These are synthetically generated, not real social media data.

#### Mode B: TwiBot-22 Benchmark

For research-grade results, the [TwiBot-22](https://github.com/DigitalHominids/TwiBot-22) dataset must be processed through `data_processing/` before it can be consumed by the core pipeline.

> **Note**: The TwiBot-22 processing pipeline (steps below) takes a **long time** to run. Pre-processed files are available for download: **[link TBD — to be added]**.

**If processing from raw data:**
```bash
cd data_processing

python twinbot_adapter_dynamic.py \
    --twibot-dir "path/to/raw/twibot22" \
    --output-dir "path/to/output" \
    --total-sample-size 1000 \
    --max-actions 50 \
    --max-follows 100

cd ..
```

This produces the standardized `twibot_{N}_v5.db` and `twibot_{N}_multimodal_v5.csv` files consumed by the core pipeline.

#### Dataset Presets

All available datasets are configured in `Character Classification/config.py`:

| `--dataset` | DB file | CSV file | Description |
|-------------|---------|----------|-------------|
| `agent72` / `72` | `data/test_72.db` | `data/72agent_deeppersonal.csv` | 72-agent demo (Mode A) |
| `twibot120` | `data/twibot_120_v5.db` | `data/twibot_120_multimodal_v5.csv` | TwiBot-120 benchmark (Mode B) |
| `twibot1000` / `twibot` | `data/twibot_1000_v5.db` | `data/twibot_1000_multimodal_v5.csv` | TwiBot-1000 benchmark (Mode B) |
| `sim1000` / `sim` | `data/simu_db/test_1000_ver2.db` | `data/simu_db/test_1000_good_bad_random_bernoulli_.csv` | 1000-agent simulation (Mode A) |

---
### Script 1: Binary Classifier (`new_main_classifier.py`)

> Extracts 26-dimensional multi-modal features (semantic PCA + behavior stats + psychology dimensions), constructs an enhanced heterogeneous graph (follow edges + cosine-similarity edges), and runs XGBoost binary bot classification with adaptive cross-validation and SHAP explainability.

Open `new_main_classifier.py` in your IDE and click the run button. Configure the `--dataset` parameter in `config.py` to select the dataset (default is `agent72`). The output folder name will be printed in the console log after execution completes.

**Output**: `new_result/hyper_newtest/{run_name}/`
```
classification_results.csv    # Per-agent predictions + probabilities
node_features.csv             # 26-dim feature vectors
enhanced_graph_edges.csv      # Graph edges (follow + cosine-similarity)
confusion_matrix.png          # Human vs Bot confusion matrix
shap_summary_plot.png         # Top-15 feature attribution beeswarm
```

---
### Script 2: Bot Gang Detection (`new_gang_detection.py`)

> Loads the bot subgraph from Script 1's output, runs Louvain community detection to identify bot gangs, and produces per-gang psychological profiling with PCA scatter visualization.

**Prerequisites**: Script 1 (`new_main_classifier.py`) must have been run **first**. This script reads the `enhanced_graph_edges.csv` and `node_features.csv` generated by Script 1. You must pass Script 1's output folder to `--save-dir`. Run from the project root directory:

```bash
# Replace the folder name with the actual one generated by Script 1
python "Character Classification/new_gang_detection.py" \
    --save-dir "new_result/hyper_newtest/classifier_agent72_test_72_20260603_110704"
```

> **Note**: The folder name contains a timestamp (e.g., `classifier_agent72_test_72_20260603_110704`). Check Script 1's console output to get the exact name, then substitute it into the command above.

**Output** (written to the same `--save-dir`):
```
gang_results.csv       # Per-agent gang assignment
gang_profiles.csv      # Per-gang psychological profile summary
bot_gang_edges.csv     # Edges within bot subgraph
gang_scatter.png       # PCA scatter colored by gang
```

---
### Script 3: Hyperbolic Role Discovery (`new_role_assigner.py`)

> Trains a Heterogeneous Graph Transformer (HGT) with Poincaré distance contrastive loss to learn structural embeddings, then runs DPMM (Dirichlet Process Mixture Model) to automatically discover tactical roles such as Opinion Leader, Information Bridge, Amplifier, Community Builder, etc.

**Prerequisites**: Script 1 (`new_main_classifier.py`) must have been run **first**. This script reads Script 1's output to build the heterogeneous graph. You must pass Script 1's output folder to `--save-dir`. Run from the project root directory:

```bash
# Replace the folder name with the actual one generated by Script 1
python "Character Classification/new_role_assigner.py" \
    --save-dir "new_result/hyper_newtest/classifier_agent72_test_72_20260603_110704"
```

> **Note**: The folder name contains a timestamp (e.g., `classifier_agent72_test_72_20260603_110704`). Check Script 1's console output to get the exact name, then substitute it into the command above.

**Output** (written to the same `--save-dir`):
```
role_assignments.csv       # user_id, role label, poincare_radius, cluster
poincare_disk.png          # Agents projected on Poincaré disk
radius_distribution.png    # Distribution of Poincaré radii
dpmm_weights.png           # DPMM cluster weights
```

---
### Script 4: Ablation Study & Independence Verification (`ablation`)

> **NEW**: Systematically validates the independent contribution of the four emotional/psychological modules (Empathy Gap, Dark Triad, Emotional Contagion, Emotion Volatility) to the final classification performance. Includes a statistical independence analysis to ensure ablation conclusions are credible.

The ablation module performs two independent analyses:

| Analysis | What It Tests | Output |
|----------|---------------|--------|
| **Independence Verification** | Whether the 4 psychological modules are statistically independent (Pearson correlation, VIF, PCA) | `independence_report.txt`, correlation heatmap, VIF bar chart, PCA scree plot |
| **Ablation Experiment** | The performance drop when each module is removed ("turn one engine off"), measured via repeated stratified cross-validation | `ablation_summary.csv`, `ablation_boxplot.png`, `ablation_heatmap.png` |

**Why ablation matters**: The classification pipeline fuses 4 psychological dimensions into a 26-dim feature vector. If these dimensions are highly collinear, removing one may not cause a performance drop — making ablation results misleading. The independence verification step ensures we *can* trust the ablation conclusions.

#### Quick Start

```bash
# Full pipeline: independence check + ablation experiment
python -m ablation.run_ablation --dataset agent72

# Independence check only (fast, skip XGBoost training)
python -m ablation.run_ablation --dataset agent72 --skip-ablation

# Ablation only (skip independence verification)
python -m ablation.run_ablation --dataset agent72 --skip-independence

# Fast test with reduced CV rounds
python -m ablation.run_ablation --dataset agent72 --repeats 2 --folds 3

# Custom data paths (bypass dataset presets)
python -m ablation.run_ablation --db "path/to/data.db" --csv "path/to/labels.csv"
```

#### Output (`ablation_results/`)

```
ablation_results/
├── independence_report.txt    # Full statistical independence report
├── correlation_heatmap.png    # Pearson correlation matrix (8-dim psycho features)
├── vif_chart.png              # Per-feature Variance Inflation Factor
├── pca_scree.png              # PCA explained variance (should show ~4 significant PCs)
├── ablation_summary.csv       # Per-module ablation performance metrics
├── ablation_boxplot.png       # Box plot of accuracy drop across CV rounds
└── ablation_heatmap.png       # Heatmap of metric changes when each module is removed
```

#### Interpretation Guide

**Independence Verification**:
- VIF < 5 per feature → modules are well-separated ✅
- Module R² < 0.5 (other 3 modules → this module) → module contributes unique signal ✅
- First 4 PCs explain ≥ 80% variance → 4 modules jointly capture most information ✅

**Ablation Results**:
- Larger accuracy drop when a module is removed → that module provides more **unique** information
- If removing a module causes no drop (or an increase), it may be redundant or noisy in isolation

---

## Supplementary Modules

These modules provide additional data, features, or visualizations for the core pipeline. They can be run independently or in sequence depending on your needs.

---
### Step A: Deep Persona Generation (`Deeppersona`)

> Uses an LLM (OpenAI/DeepSeek) to generate rich, multi-dimensional psychological profiles (demographics, career, values, lifestyle, social context, interests + narrative summary) for simulated agents.
>
> This step is based on the [Deeppersona](https://github.com/thzva/Deeppersona) framework. Download the project and run it to generate agent profiles. The output JSON file must be named `deeppersonal_agents.json` and placed in the `deeppersona_ai/` folder.
>
> A pre-generated 72-agent file is already provided at `deeppersona_ai/deeppersonal_agents.json` to skip this step.

---
### Step B: Semantic Vector Store (`deeppersona_ai`)

> Chunks personality profiles into semantic text blocks, encodes them into 384-dim vectors via SentenceTransformer, and stores in ChromaDB for fast semantic retrieval (used as RAG memory for simulation agents). This is a prerequisite for the [simulation pipeline](#step-c-multi-agent-simulation-multiagent4collusion).

**Input**: `deeppersona_ai/deeppersonal_agents.json`
**Output**: `chunked_profiles.json` + `vector_store/`

```bash
cd deeppersona_ai
python profile_chunker.py      # Profiles → semantic chunks
python build_vector_store.py   # Chunks → 384-dim ChromaDB
cd ..
```

---
### Step C: Multi-Agent Simulation (`MultiAgent4Collusion`)

> **Note**: The `data/` folder already contains a set of pre-generated simulation data (`test_72.db` + `72agent_deeppersonal.csv`). If your goal is to run the core detection pipeline (classification, role labeling, gang detection), you already have everything you need and can skip this step entirely.
>
> The simulation engine below is only needed if you want to walk through the full simulation pipeline and generate new simulation data from scratch. This is **not** the main focus of this project.

The simulation framework is adapted from **MultiAgent4Collusion** by Shanghai Jiao Tong University ([GitHub](https://github.com/renqibing/MultiAgent4Collusion)), built on top of the CAMEL-AI OASIS framework. It runs LLM-driven agents in a Twitter-like social environment where agents can post, reply, like, follow, and repost based on their personality profiles.

#### Quick Start (Skip Simulation)

If you only want the detection pipeline, use the pre-generated data already in the repository:

| File | Source |
|------|--------|
| `data/test_72.db` | Pre-built simulation database (72 agents) |
| `data/72agent_deeppersonal.csv` | Pre-built multi-modal feature CSV (72 agents) |

These files are ready to be consumed directly by the core detection pipeline (Scripts 1-3).

#### Full Simulation Pipeline

To generate new simulation data from scratch, follow these steps. Before starting, ensure you have completed [Step A](#step-a-deep-persona-generation-deeppersona) (persona generation) and [Step B](#step-b-semantic-vector-store-deeppersona_ai) (vector store) to prepare the required agent profiles.

##### Step 1: Prepare the Agent CSV

The simulation engine reads agent information from a CSV file. Use the provided `generate_simulation_csv.py` script to generate one from the Deeppersona profile JSON:

```bash
cd MultiAgent4Collusion-master
python generate_simulation_csv.py
cd ..
```

Before running, edit the configuration constants at the top of `MultiAgent4Collusion-master/generate_simulation_csv.py`:

```python
NUM_BAD_LEADER = 1      # Number of bad_leader agents
NUM_BAD_MEMBER = 1      # Number of bad_member agents
NUM_BAD = 0             # Number of bad agents (regular)
NUM_TWEETS_PER_AGENT = 5  # Seed tweets per agent
```

These three malicious agent types (`bad_leader`, `bad_member`, `bad`) are manually configured — adjust the counts based on your experiment design. The rest of the agents are automatically assigned `good`.

The script also requires tweet pool JSON files in `MultiAgent4Collusion-master/data/tweets/` (e.g., `real_tweets_COVID.json`, `fake_tweets_COVID.json`) as seed content.

**Output**: `MultiAgent4Collusion-master/our_twitter_sim/False_Business_0.csv`

The CSV contains these required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `user_id` | Unique agent ID | 0 |
| `user_char` | Personality description | "I live in Tokyo, a software engineer..." |
| `user_type` | Agent role | good / bad / bad_leader / bad_member |
| `previous_tweets` | Seed tweets (semicolon-separated) | "tweet1; tweet2" |
| `activity_level_frequency` | Activity frequency label | high / medium / low |
| `activity_level` | Numeric activity score | 0.8 |
| `username` | Display name | @User_1 |

A pre-built example is available at `MultiAgent4Collusion-master/data/simu_db/input_agents.csv`.

##### Step 2: Configure and Run the Simulation

Edit the YAML config file at `MultiAgent4Collusion-master/scripts/twitter_gpt_example/gpt_example.yaml`:

```yaml
data:
  csv_path: path/to/your_agents.csv        # Agent CSV from Step 1
  db_path: path/to/output_simulation.db    # Where to save the simulation DB
simulation:
  num_timesteps: 10                         # Number of simulation rounds
  clock_factor: 60                          # Time scaling factor
  recsys_type: twhin-bert                   # Recommendation system backend
inference:
  model_type: deepseek-chat                 # LLM model for agent reasoning
  api_key: sk-xxx                           # Your API key
  api_base_url: https://api.deepseek.com/v1 # API endpoint
```

Then launch the simulation:

```bash
cd MultiAgent4Collusion-master
python scripts/twitter_gpt_example/twitter_simulation_large.py \
    --config_path scripts/twitter_gpt_example/gpt_example.yaml
cd ..
```

**Output**: The simulation produces a SQLite database (`.db`) containing the full interaction record (users, posts, follows, likes) and can optionally export a CSV with aggregated features. These files can be consumed by the core detection pipeline.

> **⚠️ Note**: The simulation requires LLM API access (OpenAI, DeepSeek, or compatible) and can be slow for large agent counts. For quick validation of the detection pipeline, use the pre-generated data in `data/` instead.

---
### Step D: Emotional/Psychological Analysis (`emotional_analysis`)

> Four independent NLP engines loading pre-trained transformers (RoBERTa, BART, SentenceTransformer, GPT-2) to extract psychological dimensions from agent text.

| Engine | Model | Dimension |
|--------|-------|-----------|
| `EmpathyGapAnalyzer` | RoBERTa + spaCy + GPT-2 | Empathy Gap (affective × cognitive rigidity) |
| `DarkTriadAnalyzer` | BART-large-MNLI | Machiavellianism, Narcissism, Psychopathy |
| `ContagionAnalyzer` | SentenceTransformer | Emotional contagion (semantic alignment) |
| `EmotionVolatilityAnalyzer` | RoBERTa (28 emotions) | Emotion volatility (Euclidean distance) |

```python
from emotional_analysis import EmpathyGapAnalyzer, DarkTriadAnalyzer, ContagionAnalyzer, EmotionVolatilityAnalyzer

empathy = EmpathyGapAnalyzer()
result = empathy.evaluate_agent(["I lost my job and feel hopeless."])
# → {"Agent_Mean_Empathy_Gap": 0.42, ...}
```

**Note**: First run downloads ~2-5 GB of model weights from HuggingFace.

---
### Step E: Data Adapter (`data_processing`)

> Converts raw TwiBot-22 data into the standardized DB + CSV format consumed by the core pipeline.

**Input**: Raw TwiBot-22 dataset directory
**Output**: Standardized `twibot_{N}_v5.db` + `twibot_{N}_multimodal_v5.csv`

```bash
cd data_processing
python twinbot_adapter_dynamic.py \
    --twibot-dir "path/to/twibot22/data" \
    --output-dir "path/to/output" \
    --total-sample-size 1000
cd ..
```

---
### Citation

```bibtex
@inproceedings{hyperdecept2025,
  title     = {HyperDecept: A Cross-Dimensional Multimodal Framework for
               Detecting Coordinated Multi-Agent Deception},
  author    = {...},
  booktitle = {...},
  year      = {2025}
}
```
