import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests
from tqdm import tqdm

# 📡 Constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
DOWNLOAD_DIR = "./data/arxiv_pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def sanitize_filename(title: str) -> str:
    """
    Converts a paper title to a safe filename by removing special characters.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", title.strip())[:100]


def search_arxiv(
    query: str = "machine learning", max_results: int = 5
) -> List[Dict[str, str]]:
    """
    Searches the arXiv API for papers matching the given query.
    Returns a list of dicts with 'title' and 'pdf_url'.
    """
    print(f"\n🔎 Searching arXiv for: '{query}'")

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Error fetching results: {e}")
        return []

    root = ET.fromstring(response.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    results = []
    for entry in entries:
        id_url = entry.find("{http://www.w3.org/2005/Atom}id").text
        pdf_url = id_url.replace("abs", "pdf") + ".pdf"
        title = (
            entry.find("{http://www.w3.org/2005/Atom}title")
            .text.strip()
            .replace("\n", " ")
        )
        results.append({"title": title, "pdf_url": pdf_url})

    return results


def download_pdf(title: str, pdf_url: str, save_dir: str) -> None:
    """
    Downloads a PDF from arXiv and saves it using a sanitized version of the title.
    """
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"✅ Already downloaded: {filename}")
        return

    try:
        response = requests.get(pdf_url, stream=True, timeout=15)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print(f"📥 Downloaded: {filename}")

    except Exception as e:
        print(f"❌ Failed to download {pdf_url}: {e}")


def main():
    query = input("Enter arXiv search term (e.g., 'LLM retrieval'): ").strip()
    try:
        num_results = int(input("How many papers to download? "))
    except ValueError:
        print("❌ Invalid number. Please enter an integer.")
        return

    papers = search_arxiv(query=query, max_results=num_results)

    if not papers:
        print("⚠️ No results found.")
        return

    print(f"\n📚 Found {len(papers)} papers.")

    for paper in tqdm(papers, desc="Downloading"):
        print(f"\n📝 {paper['title']}")
        download_pdf(paper["title"], paper["pdf_url"], DOWNLOAD_DIR)
        time.sleep(1)  # polite pause


if __name__ == "__main__":
    main()
