"""
An ad-hoc script to add hashed id for each pair and language detection results
without reindexing the whole corpus for a week.
"""
from elasticsearch_dsl import connections
from tqdm import tqdm

from elastic_models import ParallelCorpus


if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=20)

    qs = ParallelCorpus.search()

    for doc in tqdm(qs.source(["orig", "trans"]).scan(), total=qs.count()):
        doc.hash = doc.calculate_hash()
        doc.detected_from_lang = doc.detect_language(doc.orig)
        doc.detected_to_lang = doc.detect_language(doc.trans)
        doc.save()
