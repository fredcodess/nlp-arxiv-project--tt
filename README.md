# ArXiv NLP System


An end-to-end NLP research platform for classifying and summarising scientific papers from the ArXiv repository. The system combines classical machine learning, large language models, RAG retrieval, and an interactive Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start — Streamlit App (Local)](#quick-start--streamlit-app-local)
- [Running the Jupyter Notebooks](#running-the-jupyter-notebooks)
  - [Section 1 — Data Preprocessing](#section-1--data-preprocessing)
  - [Section 2 — Classical ML](#section-2--classical-ml)
  - [Section 3 — LLM & RAG](#section-3--llm--rag)
- [Dataset Setup](#dataset-setup)
- [DeepSeek API Key Setup](#deepseek-api-key-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Key Results](#key-results)
- [Weekly Progress Summary](#weekly-progress-summary)
- [References](#references)

---

## Project Overview

The project is structured across four weekly stages:

| Week | Stage | Description |
|------|-------|-------------|
| 1 | Data Preprocessing | Load 1M ArXiv papers, filter categories, clean text, export CSV |
| 2 | Classical ML | TF-IDF + Logistic Regression / Naive Bayes / SVM, extractive summarisation, ROUGE evaluation |
| 3 | LLM & RAG | DeepSeek-V3 Zero/Few-Shot/CoT classification, FAISS-based RAG pipeline, title generation |
| 4 | Streamlit Interface | Dark-theme web app with live classification, summarisation, chatbot, and results dashboard |

**Best results achieved:**

| Task | Method | Score |
|------|--------|-------|
| Classification (1M papers) | LinearSVC + bigrams | 89.4% accuracy · 0.847 macro F1 |
| Classification (2M papers) | LinearSVC + bigrams | 92.1% accuracy · 0.879 macro F1 |
| Summarisation (classical) | TF-IDF Extractive | ROUGE-1: 0.38 |
| Summarisation (LLM RAG) | DeepSeek + RAG | ROUGE-1: 0.51 |

---

## Repository Structure

```
DG4NLP_[Your Name]/
│
├── app.py                          # Streamlit web application (Section 4)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                            # Example environment variable file
│
├── Complete.ipynb                  # Full notebook — all 4 sections
|
├── arxiv_clean.csv  
├── arxiv-metadata-oai-snapshot.json
│
├── classifier.pkl 
|
├── tfidf.pkl
|__ category_mapping.csv
```

> **Note:** The raw dataset (`arxiv-metadata-oai-snapshot.json`) is **not** included in the repository due to its size (~3GB). See [Dataset Setup](#dataset-setup) for download instructions.

---

## Requirements

### Python Version

Python **3.9 or higher** is required. The project was developed and tested on Python 3.11.

### Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
seaborn
streamlit
pandas
numpy
scikit-learn
plotly
transformers
torch
matplotlib
wordcloud
joblib
dotenv

```
---

## Quick Start — Streamlit App (Local)

The Streamlit app (`app.py`) runs **entirely standalone** — it does not require the dataset CSV or trained models. The classical ML features use a built-in heuristic TF-IDF classifier, and the LLM features connect to the DeepSeek API.

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/DG4NLP_<YourName>.git
cd DG4NLP_<YourName>
```

### Step 2 — Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure the DeepSeek API key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
DEEPSEEK_API_KEY=sk-your-key-here
```

> Get a free or paid API key at [https://platform.deepseek.com](https://platform.deepseek.com).  
> **Without the key**, all Classical ML features work fully. LLM features display a configuration notice.

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## Running the Jupyter Notebooks

The full coursework pipeline is in `Complete.ipynb`. It was developed locally in VSCode using the Jupyter extension.

### Opening in VSCode

1. Install the **Jupyter** extension in VSCode
2. Open the notebook file — VSCode renders it natively
3. Select your virtual environment as the kernel (`venv`)

---

### Section 1 — Data Preprocessing

- Loads 1,000,000 ArXiv papers in chunks of 10,000 rows
- Explores dataset structure, missing values, and category distribution
- Filters to single-label papers from the official ArXiv taxonomy (155 valid categories)
- Cleans text (whitespace, LaTeX newlines)
- Applies integer label encoding
- Exports `arxiv_clean.csv` and `category_mapping.csv`
- Generates 3 visualisation files

**Required input:** `arxiv-metadata-oai-snapshot.json` — see [Dataset Setup](#dataset-setup)

**Output files:**
```
arxiv_clean.csv
category_mapping.csv
```

---

### Section 2 — Classical ML

- Loads the cleaned CSV from Section 1
- Applies stratified sampling (min 100 / max 3,000 papers per category)
- Builds TF-IDF vectorisers in two configurations:
  - Unigrams only (50,000 features)
  - Unigrams + bigrams (100,000 features)
- Trains and evaluates 8 classification experiments:
  - Logistic Regression (C=1.0, C=5.0)
  - Naive Bayes (α=0.1, α=1.0)
  - LinearSVC (C=0.5, C=1.0)
  - SVM with bigrams
  - SVM with title×2 + abstract input
- Implements 4 extractive summarisation methods from scratch
- Computes ROUGE-1, ROUGE-2, ROUGE-L (custom implementation)
- Generates confusion matrix and 5 comparison visualisations

**Required input:** `arxiv_clean.csv`

---

### Section 3 — LLM & RAG

- Installs `openai`, `transformers`
- Connects to DeepSeek-V3 via OpenAI-compatible API
- Runs three classification prompting strategies on 60 evaluation papers:
  - Zero-Shot
  - Few-Shot (one labelled example per category)
  - Chain-of-Thought (structured reasoning + `CATEGORY: <code>` output)
- Builds a FAISS retrieval index over 5,000 papers using `all-MiniLM-L6-v2`
- Runs RAG classification and RAG summarisation
- Tests two title generation strategies (Direct, Structured)
- Generates full Classical ML vs LLM comparison charts

**Required:**
- `arxiv_clean.csv`
- `DEEPSEEK_API_KEY` configured (see below)

**Setting the API key in the notebook:**

```python
from dotenv import load_dotenv
import os
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
```

---

## Dataset Setup

The ArXiv metadata dataset is available free on Kaggle.

### Option A — Download via Kaggle CLI (recommended)

```bash
# Install kaggle CLI
pip install kaggle

# Place your kaggle.json API token at ~/.kaggle/kaggle.json
# Download the dataset
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
```

The downloaded file will be named `arxiv-metadata-oai-snapshot.json` (~3GB).

### Option B — Download via the Kaggle web interface

1. Go to [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Click **Download** (requires a free Kaggle account)
3. Unzip and place `arxiv-metadata-oai-snapshot.json` in the project root or update `DATA_PATH` in the notebook

### Update the data path in the notebook


```

---

## DeepSeek API Key Setup

### Local development — `.env` file

1. Copy the example file:
   ```bash
   cp .env .env
   ```
2. Edit `.env`:
   ```
   DEEPSEEK_API_KEY=sk-your-key-here
   ```
3. The `app.py` reads this automatically via `os.environ.get("DEEPSEEK_API_KEY")`.

### Streamlit Cloud — Secrets

When deploying to Streamlit Community Cloud, add the key through the dashboard:

1. Open your app on [https://share.streamlit.io](https://share.streamlit.io)
2. Click **⋮ → Settings → Secrets**
3. Add:
   ```toml
   DEEPSEEK_API_KEY = "sk-your-key-here"
   ```
4. The app reads this via `st.secrets["DEEPSEEK_API_KEY"]`

### Get a free or paid DeepSeek API key

Register at [https://platform.deepseek.com](https://platform.deepseek.com) — free credits are included on sign-up.

---

## Streamlit Cloud Deployment

The app is deployed from GitHub to Streamlit Community Cloud with automatic continuous deployment.

### Steps to deploy your own instance

1. **Push the repository to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [https://share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click **New app**
   - Select your repository, branch (`main`), and entry point (`app.py`)
   - Click **Deploy**

3. **Add the API key** (see [Streamlit Cloud — Secrets](#streamlit-cloud--secrets) above)

4. **Auto-deploy:** Every `git push` to `main` triggers an automatic rebuild and redeploy.

---

## Contact

Fredick Boakye · [a.htait@aston.ac.uk](mailto:a.htait@aston.ac.uk)  
Institution: Aston University, Birmingham, UK
