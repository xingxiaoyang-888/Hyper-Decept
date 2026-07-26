<<<<<<< HEAD
﻿# Hyper-Decept
=======
# Hyper-Decept
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c

HyperDecept: A cross-dimensional multimodal framework integrating LLM-native psychological profiling and hyperbolic graph learning for detecting coordinated multi-agent deception.

---

## 📋 Table of Contents

1. [Project Architecture](#-project-architecture)
2. [Prerequisites & Installation](#-prerequisites--installation)
3. [Full Pipeline Walkthrough](#-full-pipeline-walkthrough)
   - [Step 1: Generate Deep Personas (Deeppersona)](#step-1-generate-deep-personas-deeppersona)
   - [Step 2: Build Semantic Vector Store (deeppersona_ai)](#step-2-build-semantic-vector-store-deeppersona_ai)
   - [Step 3: Run Multi-Agent Simulation (MultiAgent4Collusion)](#step-3-run-multi-agent-simulation-multiagent4collusion)
   - [Step 4: Run Emotional/Psychological Analysis (emotional_analysis)](#step-4-run-emotionalpsychological-analysis-emotional_analysis)
   - [Step 5: Adapt Simulation Data (data_processing)](#step-5-adapt-simulation-data-data_processing)
   - [Step 6: Build Hyperbolic Graph & Classify Roles (Character Classification)](#step-6-build-hyperbolic-graph--classify-roles-character-classification)
   - [Step 7: Run the Ultimate Detector (main_detector.py)](#step-7-run-the-ultimate-detector-main_detectorpy)
   - [Step 8: Behavior Visualization (agent_behavior_analysis.py)](#step-8-behavior-visualization-agent_behavior_analysispy)
   - [Step 9: Advanced Visualizations (visualization/)](#step-9-advanced-visualizations-visualization)
4. [Quick Start (Minimal Demo)](#-quick-start-minimal-demo)
5. [Output Overview](#-output-overview)
6. [Citation](#-citation)
# Hyper-Decept

HyperDecept: A cross-dimensional multimodal framework integrating LLM-native psychological profiling and hyperbolic graph learning for detecting coordinated multi-agent deception.

---

## 📋 Table of Contents

1. [Project Architecture](#-project-architecture)
2. [Prerequisites & Installation](#-prerequisites--installation)
3. [Full Pipeline Walkthrough](#-full-pipeline-walkthrough)
   - [Step 1: Generate Deep Personas (Deeppersona)](#step-1-generate-deep-personas-deeppersona)
   - [Step 2: Build Semantic Vector Store (deeppersona_ai)](#step-2-build-semantic-vector-store-deeppersona_ai)
   - [Step 3: Run Multi-Agent Simulation (MultiAgent4Collusion)](#step-3-run-multi-agent-simulation-multiagent4collusion)
   - [Step 4: Run Emotional/Psychological Analysis (emotional_analysis)](#step-4-run-emotionalpsychological-analysis-emotional_analysis)
   - [Step 5: Adapt Simulation Data (data_processing)](#step-5-adapt-simulation-data-data_processing)
   - [Step 6: Build Hyperbolic Graph & Classify Roles (Character Classification)](#step-6-build-hyperbolic-graph--classify-roles-character-classification)
   - [Step 7: Run the Ultimate Detector (main_detector.py)](#step-7-run-the-ultimate-detector-main_detectorpy)
   - [Step 8: Behavior Visualization (agent_behavior_analysis.py)](#step-8-behavior-visualization-agent_behavior_analysispy)
   - [Step 9: Advanced Visualizations (visualization/)](#step-9-advanced-visualizations-visualization)
4. [Quick Start (Minimal Demo)](#-quick-start-minimal-demo)
5. [Output Overview](#-output-overview)
6. [Citation](#-citation)

---

## 🏗️ Project Architecture

```
                                HyperDecept Pipeline
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ① Deeppersona (LLM-based Personality Generation Engine)                  │
│     generate_profile.py  →  Deep multi-dimensional agent profiles        │
│              │                                                           │
│              ▼                                                           │
│  ② deeppersona_ai/ (Semantic Vector Store)                               │
│     profile_chunker.py      →  Split profiles into semantic chunks       │
│     build_vector_store.py   →  Encode → ChromaDB (384-dim vectors)       │
│              │                                                           │
│              ▼                                                           │
│  ③ MultiAgent4Collusion (Multi-Agent Simulation Engine)                  │
│     Twitter/Reddit game-theoretic simulations                             │
│     Output: SQLite DB + CSV                                              │
│              │                                                           │
│              ▼                                                           │
│  ④ emotional_analysis/ (4-Dimensional Psychological Feature Extractors)  │
│     • Empathy Gap         (Affective × Cognitive Rigidity)               │
│     • Dark Triad          (Machiavellianism / Narcissism / Psychopathy)  │
│     • Emotional Contagion (Semantic Payload Alignment)                   │
│     • Emotion Volatility  (28-D Emotion Vector Euclidean Distance)       │
│              │                                                           │
│              ▼                                                           │
│  ⑤ data_processing/ (Data Adapter Layer)                                 │
│     Standardizes simulation output → DB + CSV                            │
│              │                                                           │
│              ▼                                                           │
│  ⑥ Character Classification/ (Hyperbolic Graph + Role Detection)         │
│     graph_builder.py        →  Poincaré ball embedding                   │
│     new_role_assigner.py    →  Tactical role assignment                  │
│     new_main_detector.py    →  Fused classification                      │
│              │                                                           │
│              ▼                                                           │
│  ⑦ main_detector.py (Ultimate Tribunal)                                  │
│     Fuses: Psychology + Behavior + Semantics + Topology                  │
│     XGBoost + SHAP → Explainable bot detection                          │
│              │                                                           │
│              ▼                                                           │
│  ⑧ agent_behavior_analysis.py + visualization/                          │
│     PCA plots / KDE distributions / Radar charts / Neo4j networks        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites & Installation

### System Requirements

- **Python**: 3.10+ (recommended 3.11)
<<<<<<< HEAD
- **GPU**: CUDA-compatible GPU recommended for LLM inference and model training (CPU fallback available)
- **RAM**: 16GB+ minimum, 32GB+ recommended for large-scale simulations

### Install Core Dependencies

```bash
# Core scientific computing
pip install numpy pandas scikit-learn matplotlib seaborn

# Deep learning & transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers spacy

# Vector database
pip install chromadb

# Graph & network analysis
pip install networkx

# XGBoost & SHAP (explainable AI)
pip install xgboost shap

# Imbalanced learning
pip install imbalanced-learn

# Serialization
pip install ijson

# Neo4j visualization (optional, for dynamic follow networks)
pip install neo4j
=======
- **GPU**: CUDA-compatible GPU recommended (CPU fallback available)
- **RAM**: 16GB+ minimum, 32GB+ for large-scale simulation

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c

# Download spaCy language model
python -m spacy download en_core_web_sm
```

<<<<<<< HEAD
### Environment Variables (for LLM-based components)

```bash
# OpenAI API (for personality generation and simulation)
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE_URL="https://api.openai.com/v1"  # or your proxy

# Or for Hugging Face models
export HF_TOKEN="your-huggingface-token"
```

---

## 🚀 Full Pipeline Walkthrough

Below are the complete steps to run the entire HyperDecept pipeline from persona generation to final deception detection and visualization.

---

### Step 1: Generate Deep Personas (`Deeppersona`)

> **What**: Uses LLMs to generate rich, multi-dimensional psychological profiles for each agent (demographics, career, values, lifestyle, social context, interests + first-person narrative summary).
>
> **Input**: Attribute templates from `Deeppersona/data/`
>
> **Output**: `deeppersona_ai/deeppersonal_agents.json` — 30 agents with 6-dimensional deep personality profiles

```bash
cd Deeppersona/generate_user_profile

# Generate multiple rounds of profiles (each round generates profiles with 100-350 attributes)
python generate_profile.py

# This will:
#   1. Select random occupations and life stories from data/
#   2. For each agent, sequentially generate 6 personality dimensions via LLM
#   3. Generate a first-person narrative summary (100-400 words)
#   4. Save all profiles to deeppersona_ai/deeppersonal_agents.json

cd ../..
```

**Verification**: Open `deeppersona_ai/deeppersonal_agents.json` — you should see ~30 agents with fields like `Summary`, `Demographic Information`, `Career and Work Identity`, `Core Values, Beliefs, and Philosophy`, `Lifestyle and Daily Routine`, `Cultural and Social Context`, `Hobbies, Interests, and Lifestyle`.

---

### Step 2: Build Semantic Vector Store (`deeppersona_ai`)

> **What**: Chunks the structured personality profiles into semantic text blocks, then encodes them into 384-dimensional vectors using SentenceTransformer and stores them in ChromaDB for fast semantic retrieval.
>
> **Input**: `deeppersona_ai/deeppersonal_agents.json`
>
> **Output**: `deeppersona_ai/chunked_profiles.json` + `deeppersona_ai/vector_store/`

```bash
cd deeppersona_ai

# Step 2a: Chunk profiles into semantic blocks
python profile_chunker.py
# Output: chunked_profiles.json (~210 chunks = 30 agents × 7 sections each)
# Each chunk contains: agent_id, section (summary/demographic/career/values/...), text

# Step 2b: Build ChromaDB vector store
python build_vector_store.py
# Output: vector_store/ (persistent ChromaDB with 384-dim embeddings)
# This also runs a retrieval test with 5 sample queries to verify correctness

cd ..
```

**Verification**: After running `build_vector_store.py`, you should see retrieval results like:
```
Query: "elderly Singaporean gardening balcony chili kangkong"
  Top-2 matches:
    [agent_12_interests] distance=0.8234  agent=12 section=interests
    [agent_12_summary]   distance=0.9102  agent=12 section=summary
```

---

### Step 3: Run Multi-Agent Simulation (`MultiAgent4Collusion`)

> **What**: Runs a social network simulation where LLM-driven agents interact (post, comment, like, follow, repost) in a Twitter/Reddit-like environment. Agents are assigned `good` (human-like) or `bad` (bot/attacker) roles with distinct behavioral profiles.
>
> **Input**: YAML configuration files specifying agents, topics, and simulation parameters
>
> **Output**: SQLite database (`.db`) containing all agent actions, follow relationships, and post contents

```bash
cd MultiAgent4Collusion-master

# --- Option A: Twitter Simulation with Real-World Alignment ---
python scripts/twitter_simulation/align_with_real_world/twitter_simulation_large.py \
    --config_path scripts/twitter_simulation/align_with_real_world/yaml_200/sub1/False_Business_0.yaml

# --- Option B: Twitter Group Polarization Simulation ---
python scripts/twitter_simulation/group_polarization/twitter_simulation_group_polar.py \
    --config_path scripts/twitter_simulation/group_polarization/group_polarization.yaml

# --- Option C: Reddit Simulation Aligned with Human Behavior ---
python scripts/reddit_simulation_align_with_human/reddit_simulation_align_with_human.py \
    --config_path scripts/reddit_simulation_align_with_human/business_3600.yaml

# --- Option D: Reddit Counterfactual Content Simulation ---
python scripts/reddit_simulation_counterfactual/reddit_simulation_counterfactual.py \
    --config_path scripts/reddit_simulation_counterfactual/control_100.yaml

# --- Option E: Quick GPT-based Twitter Demo (low cost, ~33 agent inferences) ---
python scripts/twitter_gpt_example/twitter_simulation_large.py \
    --config_path scripts/twitter_gpt_example/gpt_example.yaml

cd ..
```

**Verification**: A `.db` file (e.g., `110_agent.db`) and a `.json` log file will be generated in the output directory. You can inspect them with:
```bash
python -c "import sqlite3; conn=sqlite3.connect('path/to/output.db'); \
  print('Users:', conn.execute('SELECT count(*) FROM user').fetchone()[0]); \
  print('Posts:', conn.execute('SELECT count(*) FROM post').fetchone()[0]); \
  print('Follows:', conn.execute('SELECT count(*) FROM follow').fetchone()[0])"
```

**Note for large-scale runs**: For LLM inference with open-source models, see `tutorials/installation.md` for vLLM deployment instructions.

---

### Step 4: Run Emotional/Psychological Analysis (`emotional_analysis`)

> **What**: Four independent heavy-duty NLP engines analyze each agent's tweets along four psychological dimensions. Each engine loads its own pre-trained transformer model (RoBERTa, BART, SentenceTransformer, GPT-2).
>
> **Input**: Agent tweet texts (from simulation output)
>
> **Output**: Per-agent psychological feature vectors (8 core features: mean/max for each dimension)

| Engine | Model | Dimension | What It Measures |
|---|---|---|---|
| `EmpathyGapAnalyzer` | RoBERTa + spaCy + GPT-2 | **Empathy Gap** | Affective arousal × Syntactic cognitive rigidity |
| `DarkTriadAnalyzer` | BART-large-MNLI | **Dark Triad** | Machiavellianism + Narcissism + Psychopathy via NLI |
| `ContagionAnalyzer` | SentenceTransformer | **Emotional Contagion** | Semantic alignment with manipulation payload anchors |
| `EmotionVolatilityAnalyzer` | RoBERTa (28 emotions) | **Emotion Volatility** | Euclidean distance between consecutive 28-D emotion vectors |

**Quick test** (these are singleton classes called by the feature extractor pipeline, but can be tested standalone):

```python
from emotional_analysis import EmpathyGapAnalyzer, DarkTriadAnalyzer, ContagionAnalyzer, EmotionVolatilityAnalyzer

# Initialize (each is a singleton, auto-loads models on first call)
empathy = EmpathyGapAnalyzer()
dark = DarkTriadAnalyzer()
contagion = ContagionAnalyzer()
volatility = EmotionVolatilityAnalyzer()

# Analyze a single agent's tweet history
tweets = ["I lost my job and feel hopeless.", "The system is completely rigged."]

empathy_result = empathy.evaluate_agent(tweets)
# → {"Agent_Mean_Empathy_Gap": 0.42, "Agent_Max_Empathy_Gap": 0.78, "Agent_Anomaly_Ratio": 0.15}

dark_result = dark.evaluate_agent(tweets)
# → {"Agent_Mean_Dark_Triad": 0.35, "Agent_Max_Dark_Triad": 0.82, "Agent_Manipulative_Ratio": 0.22}

contagion_result = contagion.evaluate_agent(tweets)
# → {"Agent_Mean_Alignment": 0.56, "Agent_Contagion_Spike": 0.91, "Agent_Frictionless_Index": 0.91}

volatility_result = volatility.evaluate_agent(tweets)
# → {"Agent_Mean_Volatility": 0.34, "Agent_Max_Volatility": 1.25, "Insufficient_Data": False}
```

---

### Step 5: Adapt Simulation Data (`data_processing`)

> **What**: Converts raw simulation output (or TwiBot-22 benchmark data) into a standardized format (CSV + SQLite) that downstream modules can consume. Handles text truncation, follow-graph sampling, and metadata extraction.
>
> **Input**: Raw simulation DB / TwiBot-22 dataset directory
>
> **Output**: Standardized `twibot_{N}_v5.db` + `twibot_{N}_multimodal_v5.csv`

```bash
# For TwiBot-22 benchmark data:
cd data_processing

python twinbot_adapter_dynamic.py \
    --twibot-dir "path/to/twibot22/data" \
=======
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
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c
    --output-dir "path/to/output" \
    --total-sample-size 1000 \
    --max-actions 50 \
    --max-follows 100

<<<<<<< HEAD
# Or use the V5.3 interactive version:
python twinbot_adapter.py

cd ..
```

**Output format**:
- **CSV**: `user_id`, `user_char` (bio), `followers_count`, `following_count`, `previous_tweets` (pipe-separated), `user_type` (good/bad)
- **DB Tables**: `user`, `follow`, `agent_actions` (with indexed columns for fast queries)

---

### Step 6: Build Hyperbolic Graph & Classify Roles (`Character Classification`)

> **What**: Constructs a heterogeneous graph in Poincaré ball space from the follow network, assigns tactical roles (Leader, Member, Bridge, Isolate, etc.) to each agent based on hyperbolic geometry, and performs initial multi-modal classification.
>
> **Input**: Standardized DB + CSV from Step 5
>
> **Output**: Role assignments CSV + initial classification results

```bash
cd "Character Classification"

# Step 6a: Build hyperbolic graph and compute Poincaré embeddings
python graph_builder.py

# Step 6b: Extract cross-dimensional features (semantic PCA + behavioral stats + psychology)
python new_feature_extractor.py

# Step 6c: Assign tactical roles based on hyperbolic centrality metrics
python new_role_assigner.py
# Output: ../data/hyperrole_results/hetero_hyperrole_assignments.csv

# Step 6d: Run the initial multi-modal classifier
python new_main_classifier.py

cd ..
```

**Key output**: `hetero_hyperrole_assignments.csv` contains:
- `user_id`: Agent identifier
- `Tactical_Role`: e.g., `Opinion Leader`, `Follower`, `Bridge Node`, `Isolated Node`
- `poincare_radius`: Distance from origin in Poincaré ball (high = extreme position)

---

### Step 7: Run the Ultimate Detector (`main_detector.py`)

> **What**: The "Supreme Tribunal" — fuses all cross-dimensional features (psychology + behavior stats + semantic PCA + hyperbolic topology roles) into a unified feature matrix, then trains an XGBoost classifier with adaptive cross-validation and generates SHAP-based explainability reports.
>
> **Input**:
> - `DB_FILE`: Standardized SQLite database
> - `CSV_FILE`: Multi-modal CSV with user labels
> - `ROLE_CSV`: Hyperbolic role assignments from Step 6
>
> **Output**: Classification metrics + SHAP summary plots + confusion matrix

```bash
# First, update the paths at the top of main_detector.py:
#   DB_FILE  → your simulation .db file
#   CSV_FILE → your multi-modal .csv file
#   ROLE_CSV → hetero_hyperrole_assignments.csv
#   SAVE_DIR → where results will be saved (default: results/)

python main_detector.py
```

**What happens internally**:
1. **Data Fusion Bus**: Parses tweet lists (handles string-encoded Python lists via `ast.literal_eval`), aligns user IDs across DB, CSV, and role data
2. **Feature Matrix Assembly**: Concatenates `Semantic_PCA_{i}` + `Behavior_Stat_{i}` + 8 psychology features + one-hot encoded tactical roles + Poincaré radius
3. **Adaptive Classification**:
   - **N ≤ 20**: Leave-One-Out Cross-Validation (LOOCV)
   - **N > 20**: Stratified 5-Fold CV with SMOTE oversampling
   - Automatic `scale_pos_weight` balancing for imbalanced classes
4. **SHAP Explainability**: TreeExplainer decomposes each prediction into feature contributions

**Expected output in `results/`**:
```
results/
├── confusion_matrix.png       # Confusion matrix (Human vs Bot)
├── shap_summary_plot.png      # Top-15 feature attribution beeswarm plot
└── (console output)           # Classification report + ROC-AUC score
```

---

### Step 8: Behavior Visualization (`agent_behavior_analysis.py`)

> **What**: Reads the simulation DB + CSV and generates 5 exploratory data analysis plots showing behavioral differences between `good`, `bad`, `bad_leader`, and `bad_member` agent types.
>
> **Input**: Simulation DB (e.g., `data/simu_db/yaml_200/110_agent.db`) + labeled CSV
>
> **Output**: `behavior_analysis/` directory with 5 visualization files

```bash
# First, update the paths at the top of agent_behavior_analysis.py:
#   DB_PATH  → path to your simulation .db
#   CSV_PATH → path to your labeled .csv
#   OUTPUT_DIR → "behavior_analysis"

python agent_behavior_analysis.py
```

**Output files**:
```
behavior_analysis/
├── feature_table.csv          # Per-agent numerical feature table
├── action_counts.png          # Box plots: posts, comments, likes, follows by agent type
├── pca_behavior.png           # 2D PCA projection colored by agent type
├── network_graph.png          # Directed follow-graph (top-40 nodes by in-degree)
└── feature_importance.png     # ANOVA F-score ranking of feature discriminative power
```

---

### Step 9: Advanced Visualizations (`visualization/`)

> **What**: Additional high-level visualizations for paper-quality figures.

#### 9a. Psychological Radar Charts & KDE Distributions

```bash
# Use the CognitiveVisualizer from new_visualizer.py
python new_visualizer.py
```

Generates:
- `psycho_radar_chart_tactical.png` — 8-axis radar chart of psychological fingerprints by tactical role
- `kde_tactical_Empathy_Gap_Mean.png` — KDE distribution of empathy gap per role
- `kde_tactical_Dark_Triad_Mean.png` — KDE distribution of dark triad per role
- `psycho_3d_scatter.png` — 3D psychological isolation space (Empathy × Dark Triad × Contagion)
- `shap_summary_plot.png` — SHAP feature attribution

#### 9b. Dynamic Follow Network (Neo4j)

```bash
# Prerequisites: Neo4j instance credentials in environment variables
#   NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

pip install neo4j

cd visualization/dynamic_follow_network/code

# For Reddit
python vis_neo4j_reddit.py

# For Twitter
python vis_neo4j_twitter.py

# Then open https://console.neo4j.io/ and explore user-follow-user with timestamp slicer
cd ../../..
```

#### 9c. Reddit Score Analysis

```bash
cd visualization/reddit_simulation_align_with_human/code
python analysis_all.py
# Output: Score comparison plot across treated/control/up-treated groups
cd ../../..
```

#### 9d. Twitter Group Polarization

```bash
cd visualization/twitter_simulation/group_polarization
python group_polarization_eval.py
cd ../../..
```

---

## 🔬 Quick Start (Minimal Demo)

If you want to quickly verify the entire pipeline works end-to-end without running full-scale simulations, follow this minimal path:

```bash
# 1. Generate personas (requires OpenAI API key)
cd Deeppersona/generate_user_profile && python generate_profile.py && cd ../..

# 2. Build vector store
cd deeppersona_ai && python profile_chunker.py && python build_vector_store.py && cd ..

# 3. Run a small Twitter simulation (~33 agent inferences, GPT-3.5 cost ~$0.01)
=======
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
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c
cd MultiAgent4Collusion-master
python scripts/twitter_gpt_example/twitter_simulation_large.py \
    --config_path scripts/twitter_gpt_example/gpt_example.yaml
cd ..
<<<<<<< HEAD

# 4. Run the ultimate detector with the generated DB + CSV
#    (update DB_FILE and CSV_FILE paths in main_detector.py first)
python main_detector.py

# 5. (Optional) Generate visualizations
python agent_behavior_analysis.py
python new_visualizer.py
```

---

## 📊 Output Overview

After running the full pipeline, your output structure will look like:

```
project_root/
│
├── deeppersona_ai/
│   ├── deeppersonal_agents.json      # 30 deep personality profiles
│   ├── chunked_profiles.json          # ~210 semantic chunks
│   └── vector_store/                  # ChromaDB (384-dim embeddings)
│
├── MultiAgent4Collusion-master/
│   └── data/simu_db/.../
│       └── 110_agent.db               # Simulation output
│
├── data/
│   ├── twibot_1000_v5.db              # Standardized graph database
│   ├── twibot_1000_multimodal_v5.csv  # Multi-modal feature CSV
│   └── hyperrole_results/
│       └── hetero_hyperrole_assignments.csv  # Tactical role labels
│
├── results/
│   ├── confusion_matrix.png           # Detection confusion matrix
│   └── shap_summary_plot.png          # SHAP feature attribution
│
├── behavior_analysis/
│   ├── feature_table.csv
│   ├── action_counts.png
│   ├── pca_behavior.png
│   ├── network_graph.png
│   └── feature_importance.png
│
└── new_result/
    ├── psycho_radar_chart_tactical.png
    ├── kde_tactical_*.png
    └── psycho_3d_scatter.png
```

---

## 📝 Citation

If you use HyperDecept in your research, please cite:
=======
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
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c

```bibtex
@inproceedings{hyperdecept2025,
  title     = {HyperDecept: A Cross-Dimensional Multimodal Framework for
               Detecting Coordinated Multi-Agent Deception},
  author    = {...},
  booktitle = {...},
  year      = {2025}
}
<<<<<<< HEAD
```
=======
```
>>>>>>> 4e5cd7dbb773e7dde52f6115a880e1494584129c
