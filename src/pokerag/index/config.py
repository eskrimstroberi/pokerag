"""Typed configuration for the index stage."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbedderConfig(StrictConfig):
    provider: str = "gemini"          # "gemini", "local" (fastembed/ONNX) or "st" (torch/CUDA)
    model: str = "gemini-embedding-001"
    dimension: int = 768
    task_document: str = "RETRIEVAL_DOCUMENT"
    task_query: str = "RETRIEVAL_QUERY"
    batch_size: int = 100             # Gemini limit for batchEmbedContents
    items_per_minute: int = 90        # free tier allows 100 items/minute; keep margin
    max_retries: int = 5
    timeout_seconds: int = 120
    query_prefix: str = ""            # local bge models want an instruction prefix
    device: str = "cuda"              # for provider "st"
    dtype: str = "float16"            # fp16 halves VRAM; 4 GB card
    max_seq_length: int = 512         # chunks are ~300 tokens, longer inputs waste VRAM


class QdrantConfig(StrictConfig):
    mode: str = "local"               # "local" (on-disk) or a server URL
    url: str | None = None
    distance: str = "cosine"
    on_disk: bool = True


class Bm25Config(StrictConfig):
    k1: float = 1.2
    b: float = 0.75


class IndexConfig(StrictConfig):
    snapshot: str
    strategy: str = "heading"
    artifacts_root: str = "artifacts"
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    bm25: Bm25Config = Field(default_factory=Bm25Config)

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.artifacts_root) / "corpus" / self.snapshot

    @property
    def indexes_root(self) -> Path:
        return Path(self.artifacts_root) / "indexes"


def load_index_config(path: Path) -> IndexConfig:
    """Load and validate a YAML index config."""
    return IndexConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
