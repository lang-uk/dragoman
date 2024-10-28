"""
Read sentences from TMX/XML files and calculate comet xl/xxl score
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Generator
from glob import glob
from itertools import islice

import smart_open
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
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
    print(file_path)
    input_file = smart_open.open(file_path, "r", encoding="utf-8")
    context = ET.iterparse(input_file, events=("end",))

    for _, elem in context:
        if elem.tag == "tu":
            sentence_pair = {}
            for tuv in elem.findall("tuv"):
                lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                seg = tuv.find("seg")
                if lang and seg is not None:
                    sentence_pair[lang] = seg.text.strip()

            if sentence_pair:
                yield sentence_pair
            elem.clear()


def main(
    input_dir: str,
    output_file: str,
    comet_model: str,
    src_lang: str = "en",
    tgt_lang: str = "uk",
    batch_size: int = 16,
) -> None:
    """Main function to parse XML files and measure comet score.

    Args:
        input_dir (str): The directory containing TMX/XML files.
        output_file (str): The output file in JSONL format.
        comet_model (str): The name of the comet model to use.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
    """
    all_sentences = []

    file_patterns = ["*.tmx", "*.xml", "*.xml.gz", "*.tmx.gz"]
    files = [
        file
        for pattern in file_patterns
        for file in glob(os.path.join(input_dir, "**", pattern), recursive=True)
    ]

    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)

    with open(output_file, "w", encoding="utf-8") as f:
        with tqdm(total=len(files), desc="Processing files") as pbar_files:
            pbar_files.update(1)

            with tqdm(desc="Writing sentences") as pbar_sentences:
                for file_path in files:
                    for sent_pack in batched(parse_xml_file(file_path), batch_size):
                        documents = []
                        hashes = []
                        for sent in sent_pack:
                            orig = sent[src_lang]
                            trans = sent[tgt_lang]
                            documents.append(
                                {
                                    "src": orig,
                                    "mt": trans,
                                }
                            )
                            hashes.append(calculate_hash(orig, trans))

                        model_output = model.predict(
                            documents, batch_size=batch_size, gpus=1
                        )

                        for doc, hsh, mo_score in zip(
                            documents, hashes, model_output["scores"]
                        ):
                            doc["hash"] = hsh
                            doc["comet_score"] = mo_score

                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            pbar_sentences.update(1)

                        f.flush()

    print(f"Number of files processed: {len(files)}")
    print(f"Number of sentences extracted: {len(all_sentences)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse TMX/XML files and extract sentences."
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing TMX/XML files"
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

    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_file=args.output_file,
        comet_model=args.comet_model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
    )
