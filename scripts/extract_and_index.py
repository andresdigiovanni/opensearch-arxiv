import os
import uuid
from typing import List

import pymupdf
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 📁 Configuration
PDF_DIR = "./data/arxiv_pdfs"
INDEX_NAME = "arxiv-papers"
CHUNK_SIZE = 300
EMBEDDING_DIM = 768  # for BAAI/bge-base-en

# 🔗 Connect to OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)

# 🧠 Load the embedding model
model = SentenceTransformer("BAAI/bge-base-en")


def create_index(index_name: str) -> None:
    """
    Create an OpenSearch index with vector search (k-NN) enabled.
    """
    if client.indices.exists(index=index_name):
        print(f"ℹ️ Index '{index_name}' already exists.")
        return

    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
            }
        },
        "mappings": {
            "properties": {
                "chunk_text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "source_file": {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=index_name, body=index_body)
    print(f"✅ Created index: {index_name}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract the full text from a PDF file.
    """
    try:
        doc = pymupdf.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into word-based chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > 10:  # skip very small chunks
            chunks.append(chunk)

    return chunks


def index_chunk(text_chunk: str, embedding: List[float], source_file: str) -> None:
    """
    Index a single chunk and its embedding in OpenSearch.
    """
    doc = {
        "chunk_text": text_chunk,
        "embedding": embedding,
        "source_file": source_file,
    }

    client.index(index=INDEX_NAME, body=doc, id=str(uuid.uuid4()))


def process_pdf_file(file_path: str, file_name: str) -> None:
    """
    Process a single PDF: extract text, generate embeddings, and index them.
    """
    text = extract_text_from_pdf(file_path)
    if not text:
        print(f"⚠️ No text extracted from {file_name}")
        return

    chunks = chunk_text(text)
    if not chunks:
        print(f"⚠️ No valid chunks generated from {file_name}")
        return

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for chunk, vector in zip(chunks, embeddings):
        index_chunk(chunk, vector.tolist(), file_name)


def main():
    create_index(INDEX_NAME)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    for filename in tqdm(pdf_files, desc="📄 Processing PDFs"):
        file_path = os.path.join(PDF_DIR, filename)
        process_pdf_file(file_path, filename)

    print("✅ Indexing completed.")


if __name__ == "__main__":
    main()
