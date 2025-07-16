import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

model_name = "BAAI/bge-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()


class EmbeddingRequest(BaseModel):
    input: list[str]


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    inputs = tokenizer(
        request.input, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**inputs)
        embeddings = model_output.last_hidden_state[:, 0, :].cpu().tolist()

    return {
        "data": [
            {"embedding": emb, "index": i, "object": "embedding"}
            for i, emb in enumerate(embeddings)
        ],
        "object": "list",
        "model": model_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
