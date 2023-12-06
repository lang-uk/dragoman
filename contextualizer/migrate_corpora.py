"""
An ad-hoc script to add hashed id for each pair and language detection results
without reindexing the whole corpus for a week.
"""
from elasticsearch_dsl import connections
from tqdm import tqdm

from elastic_models import ParallelCorpus, detect_language, calculate_hash
from elasticsearch_dsl import Q


if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=20)

    qs = ParallelCorpus.search()
    qs = qs.filter(~Q("exists", field="hash"))

    for doc in tqdm(qs.scan(), total=qs.count()):
        doc.hash = calculate_hash(orig=doc.orig, trans=doc.trans)
        doc.detected_from_lang = detect_language(doc.orig)
        doc.detected_to_lang = detect_language(doc.trans)
        doc.save()
