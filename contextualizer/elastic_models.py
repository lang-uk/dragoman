from typing import List
from elasticsearch_dsl import Document, Text, DenseVector, Keyword, Index, Q, Integer, Float
from elasticsearch_dsl.query import MoreLikeThis

parallel_corpus_idx = Index("parallel_corpus")


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

    @classmethod
    def get_cosine_sbert(cls, query_vector: List[float], limit: int = 8):
        return list(
            cls.search()
            .query(
                "script_score",
                script={
                    "source": "cosineSimilarity(params.queryVector, 'all_mpnet_base_v2_vector') + 1.0",
                    "params": {"queryVector": query_vector},
                },
                query=Q("match_all"),
            )[0:limit]
            .execute()
        )

    @classmethod
    def get_simple_match(cls, query: str, limit: int = 8):
        return list(cls.search().query("match", orig=query)[0:limit].execute())

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
