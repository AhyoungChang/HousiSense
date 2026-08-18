"""Lazily-loaded embedding and generation models shared across the pipeline."""

import json

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import EMBED_MODEL, GEN_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embedder = None
_generator = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    return _embedder


def embed(text):
    """Cosine-ready unit vector for a single string."""
    return embedder().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts, batch_size=256):
    return embedder().encode(texts, normalize_embeddings=True, batch_size=batch_size)


def generator():
    global _generator
    if _generator is None:
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL, device_map="auto", torch_dtype=torch.float16
        )
        _generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tok.eos_token_id,
        )
    return _generator


def complete(prompt, max_new_tokens=512):
    return generator()(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()


def complete_json(prompt, max_new_tokens=512):
    """Decode the first complete JSON object in the output and ignore any trailing text."""
    raw = complete(prompt, max_new_tokens=max_new_tokens)
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {raw[:200]}")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    return obj


def to_pgvector(vec):
    """Format a vector for a pgvector CAST(... AS vector) parameter."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
