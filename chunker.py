# backend/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    docs_dir: Path = field(default_factory=lambda: Path("documents"))
    chroma_dir: Path = field(default_factory=lambda: Path("chromadb"))
    chroma_collection: str = "smallthing_docs"
    dataset_file: Path = field(default_factory=lambda: Path("dataset_labels.jsonl"))
    conversation_log: Path = field(default_factory=lambda: Path("conversations.jsonl"))
    model_path: Path = field(default_factory=lambda: Path("classifier.pt"))
    log_file: Path = field(default_factory=lambda: Path("logs/app.log"))

    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 120

    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_results: int = 15
    similarity_threshold: float = 0.45
    rerank_top_k: int = 12

    embedding_dim: int = 384
    hidden_dims: tuple = (128, 64, 32)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16
    validation_split: float = 0.2
    min_samples_for_training: int = 20
    retrain_interval: int = 50

    max_chat_history: int = 20
    summarize_after: int = 15
    session_timeout: int = 3600

    max_file_size_mb: int = 50
    allowed_extensions: frozenset = field(default_factory=lambda: frozenset({
        ".pdf", ".txt", ".md", ".docx", ".py", ".cs", ".js", ".json", ".ts", ".cpp", ".h"
    }))
    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    api_key_enabled: bool = False
    api_key: str = field(default_factory=lambda: os.getenv("SMALLTHING_API_KEY", ""))

    log_level: str = "INFO"
    log_max_size_mb: int = 10
    log_backup_count: int = 5

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def __post_init__(self):
        for path in [self.docs_dir, self.chroma_dir, self.log_file.parent]:
            path.mkdir(parents=True, exist_ok=True)


config = Config()
