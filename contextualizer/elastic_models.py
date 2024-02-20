from typing import List
from hashlib import sha1
from elasticsearch_dsl import (
    Document,
    Text,
    DenseVector,
    Keyword,
    Index,
    Q,
    Integer,
    Float,
)
from elasticsearch_dsl.query import MoreLikeThis
import gcld3

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

parallel_corpus_idx = Index("parallel_corpus")


def calculate_hash(orig: str, trans: str) -> str:
    return sha1(f"{orig}:::{trans}".encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    result = detector.FindLanguage(text)
    return f"{result.language}-{result.is_reliable}"


@parallel_corpus_idx.document
class ParallelCorpus(Document):
    orig = Text()
    trans = Text()
    from_lang = Keyword()
    to_lang = Keyword()
    all_mpnet_base_v2_vector = DenseVector(dims=768, index=False)
    labse_orig_vector = DenseVector(dims=768, index=False)
    labse_trans_vector = DenseVector(dims=768, index=False)
    orig_len = Integer()
    trans_len = Integer()
    labse_distance = Float()
    orig_ppl = Float()
    trans_ppl = Float()
    detected_from_lang = Keyword()
    detected_to_lang = Keyword()
    hash = Keyword()

    @classmethod
    def get_cosine_sbert(cls, query_vector: List[float], limit: int = 8):
        return list(cls._cosine_sbert(query_vector=query_vector)[0:limit].execute())

    @classmethod
    def get_simple_match(cls, query: str, limit: int = 8):
        return list(cls._simple_match(query=query)[0:limit].execute())

    @classmethod
    def get_mlt(cls, query: str, limit: int = 8):
        return list(
            cls.search()
            .query(
                MoreLikeThis(
                    like=query, fields=["orig"], min_term_freq=1, max_query_terms=12
                )
            )[0:limit]
            .execute()
        )

    @classmethod
    def _cosine_sbert(cls, query_vector: List[float]) -> "ParallelCorpus.search":
        return cls.search().query(
            "script_score",
            script={
                "source": "cosineSimilarity(params.queryVector, 'all_mpnet_base_v2_vector') + 1.0",
                "params": {"queryVector": query_vector},
            },
            query=Q("match_all"),
        )

    @classmethod
    def _simple_match(cls, query: str) -> "ParallelCorpus.search":
        return cls.search().query("match", orig=query)

    @classmethod
    def _random(cls) -> "ParallelCorpus.search":
        return cls.search().query(Q("function_score", functions=[{"random_score": {}}]))
