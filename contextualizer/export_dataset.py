import pathlib
import argparse
import csv

import smart_open
from elasticsearch_dsl import connections
from tqdm import tqdm

from elastic_models import ParallelCorpus


def export_dataset(args: argparse.Namespace) -> None:
    """
    Export the dataset to a CSV file
    Args:
        args: Command line arguments
    Returns:
        None
    """
    connections.create_connection(hosts=["localhost"], timeout=20)
    qs = ParallelCorpus.search()

    fields_to_export = [
        "orig",
        "trans",
        "orig_len",
        "trans_len",
        "labse_distance",
        "orig_ppl",
        "trans_ppl",
    ]

    with smart_open.open(args.output_csv, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id"] + fields_to_export,
        )

        for i, record in enumerate(
            tqdm(qs.source(fields_to_export).scan(), total=qs.count())
        ):
            writer.writerow(
                {
                    "id": record.meta.id,
                    "orig": record.orig,
                    "trans": record.trans,
                    "orig_len": record.orig_len,
                    "trans_len": record.trans_len,
                    "labse_distance": record.labse_distance,
                    "orig_ppl": record.orig_ppl,
                    "trans_ppl": record.trans_ppl,
                }
            )

            if i and i % 10000 == 0:
                f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the dataset from a search index into a CSV file (except the vectors)"
    )
    parser.add_argument(
        "output_csv",
        type=pathlib.Path,
        help="Path to the file to export the dataset to",
    )

    args = parser.parse_args()
    export_dataset(args)
