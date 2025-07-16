build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

download:
	.venv/bin/activate
	python scripts/download_arxiv_pdfs.py

index:
	.venv/bin/activate
	python scripts/extract_and_index.py
