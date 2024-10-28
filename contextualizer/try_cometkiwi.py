"""
Read sentences from TMX/XML files and calculate comet xl/xxl score
"""

from typing import Dict, Generator, Optional
import os
import json
import argparse
import xml.etree.ElementTree as ET
from glob import glob
from itertools import islice

import smart_open
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from hashlib import sha1


def calculate_hash(orig: str, trans: str) -> str:
    return sha1(f"{orig}:::{trans}".encode("utf-8")).hexdigest()


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def parse_xml_file(file_path: str) -> Generator[Dict[str, str], None, None]:
    """Parses an XML file to extract sentences in different languages using a stream parser.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        An iterator over sentence pairs in different languages.
    """
    input_file = smart_open.open(file_path, "r", encoding="utf-8")
    context = ET.iterparse(input_file, events=("end",))

    for _, elem in context:
        if elem.tag == "tu":
            sentence_pair = {}
            for tuv in elem.findall("tuv"):
                lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                seg = tuv.find("seg")
                if lang and seg is not None and seg.text:
                    sentence_pair[lang] = seg.text.strip()

            if sentence_pair and len(sentence_pair) >= 2:
                yield sentence_pair
            elem.clear()


def main(
    input_file: str,
    output_file: str,
    comet_model: str,
    src_lang: str = "en",
    tgt_lang: str = "uk",
    batch_size: int = 16,
    start_from: Optional[int] = None,
    end_at: Optional[int] = None,
) -> None:
    """Main function to parse XML files and measure comet score.

    Args:
        input_file (str): The directory containing TMX/XML files.
        output_file (str): The output file in JSONL format.
        comet_model (str): The name of the comet model to use.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
        batch_size (int): The batch size for processing.
        start_from (Optional[int]): The index to start from.
        end_at (Optional[int]): The index to end at.
    """

    model_handle = os.path.basename(comet_model)
    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)
    curr_idx = 0

    with logging_redirect_tqdm():
        with smart_open.open(output_file, "w", encoding="utf-8") as f:
            with tqdm(desc="Writing sentences") as pbar_sentences:
                for sent_pack in batched(parse_xml_file(input_file), 100 * batch_size):
                    documents = []
                    hashes = []
                    for sent in sent_pack:
                        if start_from is not None and curr_idx < start_from:
                            curr_idx += 1
                            continue

                        curr_idx += 1
                        if end_at is not None and curr_idx >= end_at:
                            break

                        orig = sent[src_lang]
                        trans = sent[tgt_lang]
                        documents.append(
                            {
                                "src": orig,
                                "mt": trans,
                            }
                        )
                        hashes.append(calculate_hash(orig, trans))

                    if documents:
                        model_output = model.predict(
                            documents, batch_size=batch_size, gpus=1
                        )

                        for doc, hsh, mo_score in zip(
                            documents, hashes, model_output["scores"]
                        ):
                            doc["hash"] = hsh
                            doc[f"{model_handle.lower()}_score"] = mo_score

                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            pbar_sentences.update(1)

                        f.flush()

                    if end_at is not None and curr_idx >= end_at:
                        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse TMX/XML file, extract sentences and evaluate the pairs."
    )
    parser.add_argument(
        "input_file", type=str, help="Input directory containing TMX/XML files"
    )
    parser.add_argument("output_file", type=str, help="Output file in JSONL format")
    parser.add_argument(
        "--comet-model",
        type=str,
        choices=["Unbabel/wmt23-cometkiwi-da-xxl", "Unbabel/wmt23-cometkiwi-da-xl"],
        default="Unbabel/wmt23-cometkiwi-da-xxl",
    )
    parser.add_argument(
        "--src-lang", type=str, default="en", help="Source language (default: en)"
    )
    parser.add_argument(
        "--tgt-lang", type=str, default="uk", help="Target language (default: uk)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--start-from", type=int, default=None, help="Index to start from"
    )
    parser.add_argument("--end-at", type=int, default=None, help="Index to end at")

    args = parser.parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        comet_model=args.comet_model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        start_from=args.start_from,
        end_at=args.end_at,
    )
