# backend/ml/manager.py
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from backend.config import config
from backend.utils.embeddings import embedding_manager
from backend.utils.logger import logger
from backend.utils.security import SecurityManager


class SemanticClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: tuple = (128, 64, 32),
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        super().__init__()
        layers = []
        dims = (input_dim,) + hidden_dims

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DifficultyLabel(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


@dataclass
class TrainingSample:
    text: str
    label: int
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


class MLManager:
    def __init__(self):
        self.model: Optional[SemanticClassifier] = None
        self.dataset: list[TrainingSample] = []
        self.model_lock = threading.Lock()
        self.dataset_lock = threading.Lock()
        self.training_in_progress = False

    def load_model(self) -> bool:
        if not config.model_path.exists():
            logger.warning("No trained model found")
            return False
        try:
            with self.model_lock:
                self.model = SemanticClassifier(
                    input_dim=config.embedding_dim,
                    hidden_dims=config.hidden_dims,
                    dropout=config.dropout
                )
                checkpoint = torch.load(
                    config.model_path,
                    map_location="cpu",
                    weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(config.device)
                self.model.eval()
                logger.info(f"ML model loaded (trained on {checkpoint.get('num_samples', '?')} samples)")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def load_dataset(self):
        if not config.dataset_file.exists():
            return
        try:
            with self.dataset_lock:
                self.dataset = []
                with open(config.dataset_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            self.dataset.append(TrainingSample(
                                text=data["text"],
                                label=data["label"],
                                timestamp=data.get("timestamp", time.time())
                            ))
                        except (json.JSONDecodeError, KeyError):
                            continue
                logger.info(f"Loaded {len(self.dataset)} training samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")

    def save_sample(self, text: str, label: int) -> bool:
        text = SecurityManager.sanitize_text(text, max_length=5000)
        if len(text) < 10:
            return False

        sample = TrainingSample(text=text, label=label)
        try:
            with self.dataset_lock:
                self.dataset.append(sample)
                with open(config.dataset_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "text": sample.text,
                        "label": sample.label,
                        "timestamp": sample.timestamp
                    }) + "\n")

            new_count = len(self.dataset)
            if (new_count >= config.min_samples_for_training and
                    new_count % config.retrain_interval == 0):
                logger.info(f"Auto-retraining triggered ({new_count} samples)")
                threading.Thread(target=self.train, daemon=True).start()

            return True
        except Exception as e:
            logger.error(f"Failed to save sample: {e}")
            return False

    def train(self):
        if self.training_in_progress:
            logger.info("Training già in corso, skip.")
            return

        self.training_in_progress = True
        logger.info("Inizio training ML...")

        try:
            with self.dataset_lock:
                samples = list(self.dataset)

            if len(samples) < config.min_samples_for_training:
                logger.warning(f"Campioni insufficienti: {len(samples)}")
                return

            texts = [s.text for s in samples]
            labels = [s.label for s in samples]
            embeddings = embedding_manager.encode(texts, use_cache=True)

            X = torch.tensor(embeddings, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

            split = int(len(X) * (1 - config.validation_split))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            model = SemanticClassifier(
                input_dim=config.embedding_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout
            ).to(config.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            criterion = nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            best_state = None

            for epoch in range(config.epochs):
                model.train()
                optimizer.zero_grad()
                logits = model(X_train.to(config.device))
                loss = criterion(logits, y_train.to(config.device))
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val.to(config.device))
                    val_loss = criterion(val_logits, y_val.to(config.device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_state:
                torch.save({
                    "model_state_dict": best_state,
                    "num_samples": len(samples),
                    "val_loss": best_val_loss,
                    "trained_at": datetime.now().isoformat()
                }, config.model_path)

                with self.model_lock:
                    self.model = SemanticClassifier(
                        input_dim=config.embedding_dim,
                        hidden_dims=config.hidden_dims,
                        dropout=config.dropout
                    )
                    self.model.load_state_dict(best_state)
                    self.model.to(config.device)
                    self.model.eval()

                logger.info(f"Training completato — val_loss: {best_val_loss:.4f}, campioni: {len(samples)}")

        except Exception as e:
            logger.error(f"Training fallito: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_in_progress = False

    def compute_label(self, text: str) -> int:
        text_lower = text.lower()
        complexity_score = 0

        words = len(text.split())
        if words > 50:
            complexity_score += 2
        elif words > 20:
            complexity_score += 1

        technical_keywords = [
            "algorithm", "optimize", "architecture", "implement", "debug",
            "shader", "rendering", "pathfinding", "puzzle", "quest"
        ]
        technical_count = sum(1 for kw in technical_keywords if kw in text_lower)
        complexity_score += min(technical_count, 3)

        if complexity_score >= 5:
            return DifficultyLabel.HARD.value
        elif complexity_score >= 2:
            return DifficultyLabel.MEDIUM.value
        return DifficultyLabel.EASY.value

    def predict(self, text: str) -> tuple[Optional[str], Optional[float]]:
        if self.model is None:
            return None, None
        try:
            with self.model_lock:
                embedding = embedding_manager.encode_single(text)
                x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                x = x.to(config.device)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(x)
                    prob = torch.sigmoid(logits).item()

            if prob > 0.66:
                label = "HARD"
            elif prob > 0.33:
                label = "MEDIUM"
            else:
                label = "EASY"

            return label, round(prob, 4)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None

    def is_loaded(self) -> bool:
        return self.model is not None


ml_manager = MLManager()
