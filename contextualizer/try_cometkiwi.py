"""
Read sentences from TMX/XML files and calculate comet xl/xxl score
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Generator
from comet import download_model, load_from_checkpoint
from glob import glob
from tqdm import tqdm


def parse_xml_file(file_path: str) -> Generator[Dict[str, str], None, None]:
    """Parses an XML file to extract sentences in different languages.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        An iterator over sentence pairs in different languages.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    for tu in root.findall(".//tu"):
        sentence_pair = {}
        for tuv in tu.findall("tuv"):
            lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            seg = tuv.find("seg")
            if lang and seg is not None:
                sentence_pair[lang] = seg.text.strip()

        if sentence_pair:
            yield sentence_pair


def main(input_dir: str, output_file: str, comet_model: str) -> None:
    """Main function to parse XML files and measure comet score.

    Args:
        input_dir (str): The directory containing TMX/XML files.
        output_file (str): The output file in JSONL format.
    """
    all_sentences = []
    files = glob(os.path.join(input_dir, "**", "*.tmx"), recursive=True) + glob(
        os.path.join(input_dir, "**", "*.xml"), recursive=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        with tqdm(total=len(files), desc="Processing files") as pbar_files:
            pbar_files.update(1)

            with tqdm(desc="Writing sentences") as pbar_sentences:
                for file_path in files:
                    for sent in parse_xml_file(file_path):
                        print(sent)
                        pbar_sentences.update(1)
                        raise Exception("Stop here")


            # for sentence in all_sentences:
            #     f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
            #     pbar_sentences.update(1)

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
        "comet-model",
        type=str,
        choices=["Unbabel/wmt23-cometkiwi-da-xxl", "Unbabel/wmt23-cometkiwi-da-xl"],
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.comet_model)
