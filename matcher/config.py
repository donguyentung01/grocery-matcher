from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchConfig:
    store_a: Path
    store_b: Path
    output: Path
    cache_dir: Path
    credentials: Path
    embedding_model: str
    lexical_candidates: int
    vector_candidates: int
    lexical_min_score: float
    vector_min_score: float
    max_gpt_calls: int
    gpt_concurrency: int


def parse_args() -> MatchConfig:
    parser = argparse.ArgumentParser(
        description="Match Store A products to validated Store B equivalents."
    )
    parser.add_argument("--store-a", type=Path, required=True)
    parser.add_argument("--store-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("openai_creds.yaml"),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--lexical-candidates", type=int, default=5)
    parser.add_argument("--vector-candidates", type=int, default=5)
    parser.add_argument("--lexical-min-score", type=float, default=0.15)
    parser.add_argument("--vector-min-score", type=float, default=0.50)
    parser.add_argument("--max-gpt-calls", type=int, default=250_000)
    parser.add_argument("--gpt-concurrency", type=int, default=50)
    args = parser.parse_args()

    for label, value in (
        ("lexical-candidates", args.lexical_candidates),
        ("vector-candidates", args.vector_candidates),
        ("max-gpt-calls", args.max_gpt_calls),
        ("gpt-concurrency", args.gpt_concurrency),
    ):
        if value < 1:
            parser.error(f"--{label} must be positive")
    for label, value in (
        ("lexical-min-score", args.lexical_min_score),
        ("vector-min-score", args.vector_min_score),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{label} must be between 0 and 1")
    return MatchConfig(
        store_a=args.store_a.resolve(),
        store_b=args.store_b.resolve(),
        output=args.output.resolve(),
        cache_dir=args.cache_dir.resolve(),
        credentials=args.credentials.resolve(),
        embedding_model=args.embedding_model,
        lexical_candidates=args.lexical_candidates,
        vector_candidates=args.vector_candidates,
        lexical_min_score=args.lexical_min_score,
        vector_min_score=args.vector_min_score,
        max_gpt_calls=args.max_gpt_calls,
        gpt_concurrency=args.gpt_concurrency,
    )
