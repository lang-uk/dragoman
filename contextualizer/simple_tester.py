from elastic_models import ParallelCorpus
from elasticsearch_dsl import connections
from sentence_transformers import SentenceTransformer


connections.create_connection(hosts=["localhost"], timeout=120)
model = SentenceTransformer("all-mpnet-base-v2")

query = "Weather in Berlin: cold, cloudy and windy"
embeddings = list(model.encode(query))


print("Query: ", query)

print("\n\nCosine similarity on sbert:")
print("===========================")
for res in ParallelCorpus.get_cosine_sbert(embeddings):
    print(res.orig)

print("\n\nSimple match on BM25 (or whatever):")
print("===================================")
for res in ParallelCorpus.get_simple_match(query):
    print(res.orig)


print("\n\nMoreLikeThis on BM25 (or whatever):")
print("===================================")
for res in ParallelCorpus.get_mlt(query):
    print(res.orig)
