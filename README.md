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

# Download spaCy language model
python -m spacy download en_core_web_sm
```

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
    --output-dir "path/to/output" \
    --total-sample-size 1000 \
    --max-actions 50 \
    --max-follows 100

# Or use the V5.3 interactive version:
python twinbot_adapter.py

cd ..
```

**Output format**:
- **CSV**: `user_id`, `user_char` (bio), `followers_count`, `following_count`, `previous_tweets` (pipe-separated), `user_type` (good/bad)
- **DB Tables**: `user`, `follow`, `agent_actions` (with indexed columns for fast queries)

---

### Step 6: Multi-Modal Classification & Role Discovery (`Character Classification`)

> **What**: A three-script pipeline that extracts multi-modal features (semantic + behavioral + psychological), builds an enhanced heterogeneous graph with cosine-similarity edges, runs XGBoost binary bot classification (Script 1), detects bot gang communities (Script 2), and discovers tactical roles via HGT + Poincaré hyperbolic projection + DPMM clustering (Script 3).
>
> **Input**: Standardized DB + CSV from Step 5
>
> **Output**: Classification reports, gang detection results, and hyperbolic role assignments

**Script 1 — Classifier (required, runs first):**

```bash
cd "Character Classification"

# Build 26-dim multi-modal features (semantic PCA + behavior stats + 4 psychology dimensions)
# → Construct enhanced heterogeneous graph (follow edges + cosine-similarity edges)
# → XGBoost binary classification with adaptive CV (LOOCV for ≤20, 5-Fold SMOTE for larger)
# → SHAP feature attribution + confusion matrix
python new_main_classifier.py --dataset agent72
# Output: new_result/hyper_newtest/classification_results.csv
#         new_result/hyper_newtest/node_features.csv
#         new_result/hyper_newtest/enhanced_graph_edges.csv
#         new_result/hyper_newtest/confusion_matrix.png
#         new_result/hyper_newtest/shap_summary_plot.png

cd ..
```

Available datasets (from `config.py` presets, or use `--db`/`--csv` for custom paths):

| `--dataset` | DB file | CSV file | Description |
|---|---|---|---|
| `agent72` / `72` | `data/test_72.db` | `data/72agent_deeppersonal.csv` | 72-agent small demo |
| `twibot120` | `data/twibot_120_v5.db` | `data/twibot_120_multimodal_v5.csv` | TwiBot-120 benchmark |
| `twibot1000` / `twibot` | `data/twibot_1000_v5.db` | `data/twibot_1000_multimodal_v5.csv` | TwiBot-1000 benchmark |
| `sim1000` / `sim` | `data/simu_db/test_1000_ver2.db` | `data/simu_db/test_1000_good_bad_random_bernoulli_.csv` | 1000-agent simulation |

**Script 2 — Bot Gang Detection (optional, runs after Script 1):**

> **Note**: Script 1 saves outputs to a timestamped subfolder under `{save_dir}` (e.g., `new_result/hyper_newtest/classifier_agent72_.../`). You must pass this folder path to `--save-dir` so Script 2 can find the required files.

```bash
cd "Character Classification"

# Unsupervised bot community detection on the enhanced graph
# → Loads bot subgraph from Script 1's enhanced_graph_edges.csv
# → Louvain community detection → gang assignments
# → Per-gang psychological profiling + PCA scatter visualization
python new_gang_detection.py --save-dir "{Script1_output_dir}"
# Output: {save_dir}/gang_results.csv
#         {save_dir}/gang_profiles.csv
#         {save_dir}/gang_scatter.png

cd ..
```

**Script 3 — Hyperbolic Role Discovery (optional, runs after Script 1):**

> **Note**: Same as Script 2 — pass the Script 1 output folder to `--save-dir`.

```bash
cd "Character Classification"

# HGT (Heterogeneous Graph Transformer) + Poincaré ball projection + DPMM
# → Learns structural embeddings via Poincaré distance contrastive loss
# → Automatically discovers tactical roles (Opinion Leader, Information Bridge, etc.)
# → Visualizes agents on Poincaré disk with role-based coloring
python new_role_assigner.py --save-dir "{Script1_output_dir}"
# Output: {save_dir}/role_assignments.csv
#         {save_dir}/poincare_disk.png
#         {save_dir}/radius_distribution.png

cd ..
```

**Template for full Step 6 workflow:**
```bash
# 1. First, list available dataset presets:
python "Character Classification/new_main_classifier.py" --help

# 2. Run Script 1 (classifier) — pick your dataset, outputs go to a timestamped folder:
cd "E:\Hyper-Decept\Hyper-Decept"
python "Character Classification/new_main_classifier.py" --dataset agent72

# 3. Find the output folder name (printed in the log, e.g. "Output: .../classifier_agent72_20260602_123456"):
#    Then pass it as --save-dir to Script 2 and 3:
python "Character Classification/new_gang_detection.py" --save-dir "new_result/hyper_newtest/classifier_agent72_20260602_123456"
python "Character Classification/new_role_assigner.py" --save-dir "new_result/hyper_newtest/classifier_agent72_20260602_123456"
```

**Key output**: `role_assignments.csv` contains:
- `user_id`: Agent identifier
- `role`: e.g., `Opinion Leader`, `Information Bridge`, `Amplifier`, `Community Builder`, `Peripheral`, `Fringe Node`, `Boundary Node`, `Outsider`
- `poincare_radius`: Distance from origin in Poincaré ball (smaller = more central)
- `cluster`: DPMM cluster index
- `user_type` / `is_bad`: Ground truth labels (merged if available)

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
cd MultiAgent4Collusion-master
python scripts/twitter_gpt_example/twitter_simulation_large.py \
    --config_path scripts/twitter_gpt_example/gpt_example.yaml
cd ..

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

```bibtex
@inproceedings{hyperdecept2025,
  title     = {HyperDecept: A Cross-Dimensional Multimodal Framework for
               Detecting Coordinated Multi-Agent Deception},
  author    = {...},
  booktitle = {...},
  year      = {2025}
}
```