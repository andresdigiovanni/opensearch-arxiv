# 🔎 Semantic Search on arXiv PDFs with OpenSearch & BGE Embeddings

This project sets up a full pipeline for semantic search over scientific papers:

* 📄 Downloads papers from **arXiv**
* 🧠 Extracts text and computes embeddings using **BAAI/bge-base-en** locally
* 🔍 Indexes both dense and sparse (BM25) fields into **OpenSearch**
* 🔁 Performs **BM25**, **vector (KNN)**, and **hybrid search** queries
* 🧪 Interactively tested via a Jupyter notebook

---

## 🚀 Features

* 🔨 Local embedding using HuggingFace model (`BAAI/bge-base-en`)
* 📦 Dockerized stack for **OpenSearch** and **Dashboards**
* 🧾 arXiv PDF downloader and text chunker
* ⚙️ Fast, semantic indexing with dense vector support
* 🔍 Hybrid retrieval combining BM25 + vector similarity
* 🧠 No external API or LiteLLM dependency — all local

---

## 🧱 Architecture

```
arXiv PDFs → text chunks → BGE embeddings (locally via HuggingFace)
                     ↓
              Index in OpenSearch
                     ↓
     Query: BM25 / Vector / Hybrid (via notebook)
```

---

## 📁 Project Structure

```
opensearch-arxiv/
├── embedding-api/                # Embedding microservice (optional, unused in CLI)
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   └── search_examples.ipynb     # BM25, vector, and hybrid queries
├── scripts/
│   ├── download_arxiv_pdfs.py    # Download arXiv PDFs
│   └── extract_and_index.py      # Chunk, embed, and index
├── data/
│   └── arxiv_pdfs/               # Downloaded PDFs
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## ⚙️ Requirements

* Docker + Docker Compose (for OpenSearch)
* Python 3.11+ (for running scripts locally)
* [`uv`](https://github.com/astral-sh/uv) as the Python package manager
* 8GB+ RAM recommended for local embeddings

---

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/andresdigiovanni/opensearch-arxiv.git
cd opensearch-arxiv
```

### 2. Build Docker containers

```bash
make build
```

### 3. Start OpenSearch

```bash
make up
```

This launches:

* OpenSearch at `http://localhost:9200`
* OpenSearch Dashboards at `http://localhost:5601`

---

## 📥 Download arXiv PDFs

```bash
make download
```

This downloads a small sample of arXiv PDFs into `data/arxiv_pdfs/`.

> You can edit `scripts/download_arxiv_pdfs.py` to customize topics, queries or paper IDs.

---

## 🧠 Embed and Index into OpenSearch

```bash
make index
```

This script will:

* Extract and chunk text from each PDF
* Compute sentence embeddings using `BAAI/bge-base-en`
* Index into OpenSearch with:

  * Sparse field (`text`) for BM25
  * Dense field (`embedding`) for k-NN vector search

---

## 🔎 Run Search Queries (Notebook)

Start Jupyter and open the notebook:

```bash
jupyter notebook notebooks/search_examples.ipynb
```

Inside you'll find:

* ✅ BM25 search examples
* 📐 k-NN (dense vector) search
* 🔁 Hybrid queries combining both

---

## 🧰 Makefile Commands

```makefile
make build     # Build Docker services
make up        # Start OpenSearch and Dashboards
make down      # Stop all containers
make download  # Download PDFs from arXiv
make index     # Chunk, embed, and index into OpenSearch
```

---

## ❌ What was removed?

Originally, LiteLLM was considered to expose `/v1/embeddings` via an OpenAI-compatible API.
However, this setup now **uses local HuggingFace inference directly**, without needing LiteLLM or external APIs.

---

## 📚 References

* [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)
* [OpenSearch k-NN plugin](https://opensearch.org/docs/latest/search-plugins/knn/)
* [arXiv API](https://arxiv.org/help/api/)

---

## 👤 Author

Developed by [Andrés Di Giovanni](https://github.com/andresdigiovanni), 2025.
