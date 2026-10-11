from __future__ import annotations

import hashlib

import numpy as np


class FrozenOutputEmbedderV8:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        text = str(text or "")
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.cache:
            v = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            self.cache[key] = np.asarray(v, dtype=np.float32)
        return self.cache[key]
