"""
Checks sentences in TMX/XML files for sanity
"""

import os
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


def main(input_dir: str) -> None:
    """Main function to parse XML files and check sentence lenghts

    Args:
        input_dir (str): The directory containing TMX/XML files.
        output_file (str): The output file in JSONL format.
    """
    files = glob(os.path.join(input_dir, "**", "*.tmx"), recursive=True) + glob(
        os.path.join(input_dir, "**", "*.xml"), recursive=True
    )

    with tqdm(total=len(files), desc="Processing files") as pbar_files:
        for file_path in files:
            sentences = parse_xml_file(file_path)
            for i, sentence in enumerate(sentences):
                lens = list(map(len, sentence.values()))
                for text in sentence.values():
                    if len(text) > 500:
                        print(f"Sentence too long, File: {file_path}:{i}, Len: {len(text)} Text: {text}")
            
                if max(lens) - min(lens) > 100:
                    print(f"Language length difference too large, File: {file_path}:{i}, Lenghts: {lens}")

            pbar_files.update(1)


    print(f"Number of files processed: {len(files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse TMX/XML files and extract sentences."
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing TMX/XML files"
    )

    args = parser.parse_args()
    main(args.input_dir)
