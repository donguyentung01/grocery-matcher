from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from matcher.models import Candidate, ProductRecord


def _cache_key(records: list[ProductRecord], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(str(len(records)).encode())
    for record in records[:100]:
        digest.update(record.item_id.encode())
        digest.update(record.retrieval_text.encode())
    if records:
        digest.update(records[-1].item_id.encode())
    return digest.hexdigest()[:16]


class CandidateRetriever:
    def __init__(
        self,
        b_records: list[ProductRecord],
        cache_dir: Path,
        embedding_model: str,
        lexical_candidates: int,
        vector_candidates: int,
        lexical_min_score: float,
        vector_min_score: float,
    ) -> None:
        self.b_records = b_records
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model
        self.lexical_candidates = lexical_candidates
        self.vector_candidates = vector_candidates
        self.lexical_min_score = lexical_min_score
        self.vector_min_score = vector_min_score
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.name_index: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(b_records):
            self.name_index[record.normalized_name].append(index)

        self.vectorizer: TfidfVectorizer | None = None
        self.b_lexical: csr_matrix | None = None
        self.embedding_model: SentenceTransformer | None = None
        self.faiss_index: faiss.Index | None = None
        self._prepare_lexical()
        self._prepare_embeddings()

    def _prepare_lexical(self) -> None:
        texts = [record.retrieval_text for record in self.b_records]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            max_features=300_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        self.b_lexical = self.vectorizer.fit_transform(texts).tocsr()

    def _prepare_embeddings(self) -> None:
        key = _cache_key(self.b_records, self.embedding_model_name)
        vector_path = self.cache_dir / f"store_b_{key}.npy"
        index_path = self.cache_dir / f"store_b_{key}.faiss"

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        if index_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))
        else:
            if vector_path.exists():
                vectors = np.load(vector_path)
            else:
                vectors = self.embedding_model.encode(
                    [record.retrieval_text for record in self.b_records],
                    batch_size=256,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                np.save(vector_path, vectors)
            dimension = vectors.shape[1]
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = max(64, self.vector_candidates * 2)
            index.add(vectors)
            faiss.write_index(index, str(index_path))
            self.faiss_index = index

    def _add_exact(
        self, a_record: ProductRecord, candidate_map: dict[int, Candidate]
    ) -> None:
        for b_index in self.name_index.get(a_record.normalized_name, ()):
            candidate = candidate_map.setdefault(
                b_index, Candidate(b_index=b_index)
            )
            candidate.exact_name = True

    def retrieve_batch(self, a_records: list[ProductRecord]) -> list[list[Candidate]]:
        if self.vectorizer is None or self.b_lexical is None:
            raise RuntimeError("Lexical index is not prepared")

        candidate_maps: list[dict[int, Candidate]] = [dict() for _ in a_records]
        a_texts = [record.retrieval_text for record in a_records]
        a_lexical = self.vectorizer.transform(a_texts).tocsr()
        
        lexical = sp_matmul_topn(
            a_lexical,
            self.b_lexical.T,
            top_n=self.lexical_candidates,
            threshold=self.lexical_min_score,
            sort=True,
            n_threads=max(1, os.cpu_count() or 1),
        )

        for local_index in range(lexical.shape[0]):
            row = lexical.getrow(local_index)
            for b_index, score in zip(row.indices, row.data, strict=True):
                candidate_maps[local_index][int(b_index)] = Candidate(
                    b_index=int(b_index),
                    lexical_score=float(score),
                )

        if self.embedding_model is None or self.faiss_index is None:
            raise RuntimeError("Embedding index is not prepared")
        a_vectors = self.embedding_model.encode(
            a_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        vector_scores, vector_indices = self.faiss_index.search(
            a_vectors, self.vector_candidates
        )
        for local_index, (indices, scores) in enumerate(
            zip(vector_indices, vector_scores, strict=True)
        ):
            for b_index, score in zip(indices, scores, strict=True):
                if b_index < 0 or score < self.vector_min_score:
                    continue
                candidate = candidate_maps[local_index].setdefault(
                    int(b_index),
                    Candidate(b_index=int(b_index)),
                )
                candidate.vector_score = max(candidate.vector_score, float(score))

        output: list[list[Candidate]] = []
        for a_record, candidate_map in zip(a_records, candidate_maps, strict=True):
            self._add_exact(a_record, candidate_map)
            output.append(list(candidate_map.values()))
        return output
