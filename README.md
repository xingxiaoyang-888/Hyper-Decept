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