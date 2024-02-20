import argparse
import csv
import json
from tqdm import tqdm
from elasticsearch_dsl import connections
from sentence_transformers import SentenceTransformer

from elastic_models import ParallelCorpus


def main(args):
    connections.create_connection(hosts=["localhost"], timeout=120)

    if args.method == "sbert":
        model = SentenceTransformer("all-mpnet-base-v2")

    csv_reader = csv.DictReader(args.input_file)

    for row in tqdm(csv_reader, desc="Processing"):
        query = row[args.eng_field_name]

        if args.method == "sbert":
            embeddings = list(model.encode(query))
            queryset = ParallelCorpus._cosine_sbert(embeddings)
        elif args.method == "bm25":
            queryset = ParallelCorpus._simple_match(query)
        else:  # random
            queryset = ParallelCorpus._random()

        if args.filter == "strict":
            queryset = (
                queryset.filter("term", detected_to_lang="uk-True")
                .filter("term", detected_from_lang="en-True")
                .filter("range", labse_distance={"gt": 0.91})
                .filter("range", orig_len={"lte": 1024})
                .filter(
                    "script",
                    script={
                        "lang": "painless",
                        "source": "(doc['orig_ppl'].value + doc['trans_ppl'].value) < 3.33",
                    },
                )
                .filter(
                    "script",
                    script={
                        "lang": "painless",
                        "source": "Math.abs(doc['orig_len'].value - doc['trans_len'].value) < 50",
                    },
                )
                .source(True)
            )
        elif args.filter == "loose":
            queryset = (
                queryset.filter("term", detected_to_lang="uk-True")
                .filter("term", detected_from_lang="en-True")
                .filter("range", labse_distance={"gt": 0.85})
                .filter("range", orig_len={"lte": 1024})
                .filter(
                    "script",
                    script={
                        "lang": "painless",
                        "source": "(doc['orig_ppl'].value + doc['trans_ppl'].value) < 3.25",
                    },
                )
                .filter(
                    "script",
                    script={
                        "lang": "painless",
                        "source": "Math.abs(doc['orig_len'].value - doc['trans_len'].value) < 50",
                    },
                )
                .source(True)
            )

        results = list(queryset[0 : args.first_n].execute())

        args.output_file.write(
            json.dumps(
                {
                    "orig": query,
                    "trans": row[args.ukr_field_name],
                    "context": [
                        {
                            "orig": res.orig.strip(),
                            "trans": res.trans.strip(),
                        }
                        for res in results
                    ],
                },
                ensure_ascii=False,
            ) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export contextualized data from Elasticsearch"
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="CSV with the to find similar samples to",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="JSONL to write the output to",
    )

    parser.add_argument(
        "--eng_field_name",
        default="sentence_eng_Latn",
        help="CSV fieldname for the English sentence",
    )
    parser.add_argument(
        "--ukr_field_name",
        default="sentence_ukr_Cyrl",
        help="CSV fieldname for the Ukrainian sentence",
    )

    parser.add_argument(
        "--filter",
        choices=["none", "loose", "strict"],
        help="Filter out documents in the similar way to the export_*_corpus.py scripts"
        "where loose is 3m and strict is 1m",
    )

    parser.add_argument(
        "--method",
        choices=["sbert", "bm25", "random"],
        default="sbert",
    )
    parser.add_argument(
        "--first_n",
        type=int,
        default=10,
        help="First n similar documents to return",
    )

    args = parser.parse_args()
    main(args)
