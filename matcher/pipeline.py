from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

from matcher.config import MatchConfig, parse_args
from matcher.gpt import GptAdjudicator
from matcher.io import load_products
from matcher.models import Candidate
from matcher.normalize import enrich_missing_brands
from matcher.retrieval import CandidateRetriever


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    config: MatchConfig,
    a_records,
    b_records,
    selected: dict[int, int],
) -> None:
    _ensure_parent(config.output)

    ordered = sorted(selected.items(), key=lambda item: a_records[item[0]].item_id)
    with config.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id_A", "name_A", "item_id_B", "name_B"])
        for a_index, b_index in ordered:
            writer.writerow(
                [
                    a_records[a_index].item_id,
                    a_records[a_index].raw_name,
                    b_records[b_index].item_id,
                    b_records[b_index].raw_name,
                ]
            )

    submission_path = config.output.with_name(
        f"{config.output.stem}_submission{config.output.suffix}"
    )
    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id_A", "item_id_B"])
        for a_index, b_index in ordered:
            writer.writerow([a_records[a_index].item_id, b_records[b_index].item_id])

async def run(config: MatchConfig) -> dict:
    started = time.time()
    a_records = load_products(config.store_a)
    b_records = load_products(config.store_b)
    if not a_records or not b_records:
        raise ValueError("Both datasets must contain valid products")
    enrich_missing_brands((a_records, b_records))

    retriever = CandidateRetriever(
        b_records=b_records,
        cache_dir=config.cache_dir,
        embedding_model=config.embedding_model,
        lexical_candidates=config.lexical_candidates,
        vector_candidates=config.vector_candidates,
        lexical_min_score=config.lexical_min_score,
        vector_min_score=config.vector_min_score,
    )
    adjudicator = GptAdjudicator(
        config.credentials,
        config.cache_dir,
        config.gpt_concurrency,
    )

    selected: dict[int, int] = {}
    batch_size = 1024
    for offset in range(0, len(a_records), batch_size):
        batch = a_records[offset : offset + batch_size]
        candidate_lists = retriever.retrieve_batch(batch)
        pending_gpt: list[tuple[int, list[Candidate]]] = []
        for local_index, candidates in enumerate(candidate_lists):
            a_index = offset + local_index
            if not candidates:
                continue
            pending_gpt.append((a_index, candidates))

        if pending_gpt:
            adjudications = await asyncio.gather(
                *(
                    adjudicator.adjudicate(
                        a_records[a_index],
                        b_records,
                        candidates,
                        config.max_gpt_calls,
                    )
                    for a_index, candidates in pending_gpt
                )
            )
            for (a_index, _), b_index in zip(
                pending_gpt,
                adjudications,
                strict=True,
            ):
                if b_index is None:
                    continue
                selected[a_index] = b_index
        print(
            f"Processed {min(offset + batch_size, len(a_records)):,}/{len(a_records):,}; "
            f"accepted={len(selected):,}; gpt_calls={adjudicator.calls:,}; "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    stats = {
        "processed_a": len(a_records),
        "final_matches": len(selected),
        "elapsed_seconds": round(time.time() - started, 3),
        "embedding_model": config.embedding_model,
        "gpt_calls": adjudicator.calls,
        "gpt_cache_hits": adjudicator.cache_hits,
        "gpt_concurrency": config.gpt_concurrency,
        "lexical_min_score": config.lexical_min_score,
        "vector_min_score": config.vector_min_score,
    }
    _write_outputs(config, a_records, b_records, selected)
    return stats


def main() -> None:
    config = parse_args()
    stats = asyncio.run(run(config))
    print(json.dumps(stats, indent=2))
