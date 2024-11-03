"""
Export sentences from TMX/XML files to a JSONL file for all the languages
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict
from glob import glob
from tqdm import tqdm


def parse_xml_file(file_path: str) -> List[Dict[str, str]]:
    """Parses an XML file to extract sentences in different languages.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains
        sentences in different languages.
    """
    sentences = []
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
            sentences.append(sentence_pair)

    return sentences


def main(input_dir: str, output_file: str) -> None:
    """Main function to parse XML files and write sentences to a JSONL file.

    Args:
        input_dir (str): The directory containing TMX/XML files.
        output_file (str): The output file in JSONL format.
    """
    all_sentences = []
    files = glob(os.path.join(input_dir, "**", "*.tmx"), recursive=True) + glob(
        os.path.join(input_dir, "**", "*.xml"), recursive=True
    )

    with tqdm(total=len(files), desc="Processing files") as pbar_files:
        for file_path in files:
            sentences = parse_xml_file(file_path)
            all_sentences.extend(sentences)
            pbar_files.update(1)

    with open(output_file, "w", encoding="utf-8") as f:
        with tqdm(total=len(all_sentences), desc="Writing sentences") as pbar_sentences:
            for sentence in all_sentences:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
                pbar_sentences.update(1)

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

    args = parser.parse_args()
    main(args.input_dir, args.output_file)
