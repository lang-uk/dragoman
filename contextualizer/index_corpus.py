from typing import List, Dict, Iterable, Generator
import argparse
import pathlib
import logging
import math
from itertools import islice

from tqdm import tqdm
import smart_open
import torch
from elastic_models import ParallelCorpus, parallel_corpus_idx
from elasticsearch.helpers import streaming_bulk
from elasticsearch.client import Elasticsearch
from elasticsearch_dsl import connections
from sentence_transformers import SentenceTransformer, util as sbert_utils
from transformers import AutoTokenizer, AutoModelForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batched(iterable: Iterable, n: int) -> Generator[List, None, None]:
    """
    Yield batches of size n from iterable
    Args:
        iterable: Iterable to batch
        n: Batch size
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def bulk_write(conn: Elasticsearch, docs_to_index: List[Dict]) -> None:
    """
    Index a batch of documents
    Args:
        conn: Elasticsearch connection
        docs_to_index: List of documents to index
    Returns:
        None
    """
    if docs_to_index:
        for _ in streaming_bulk(conn, docs_to_index):
            pass


def get_bpc(model, tokenizer, text: str, device: str) -> float:
    """
    Get bits per character for a given text
    Args:
        model: Causal language model
        tokenizer: Tokenizer for the model
        text: Text to estimate perplexity for
        device: Device to use for the model
    """
    x = tokenizer.encode(text, add_special_tokens=False)
    x = torch.LongTensor([tokenizer.eos_token_id] + x)

    x = x.to(device).long()

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cpu"):
            y = model(input_ids=x[None, :]).logits

    x = x[1:]
    y = y[0, : x.size(-1), :]
    log_prob_per_token = torch.nn.functional.cross_entropy(y, x)
    log_prob = log_prob_per_token.item() * x.size(-1)

    return log_prob / math.log(2) / len(text)


def main(args):
    logger.info(f"Using device {args.device}")

    connections.create_connection(hosts=["localhost"], timeout=20)

    if args.drop_index:
        logger.warning("Dropping the index, because you said so")
        parallel_corpus_idx.delete(ignore=404)

    logger.info("Loading sbert model for similarity search")
    model = SentenceTransformer(args.transformer_model, device=args.device)
    logger.info("Loading LaBSE model for translation quality estimation")
    labse_model = SentenceTransformer(args.labse_model, device=args.device)

    logger.info("Loading causal model to estimate perplexity in original language")
    orig_ppl_model = AutoModelForCausalLM.from_pretrained(args.orig_ppl_model)
    orig_ppl_model.to(args.device)
    orig_ppl_tokenizer = AutoTokenizer.from_pretrained(args.orig_ppl_model)
    logger.info("Loading causal model to estimate perplexity in translated language")
    trans_ppl_model = AutoModelForCausalLM.from_pretrained(args.trans_ppl_model)
    trans_ppl_model.to(args.device)
    trans_ppl_tokenizer = AutoTokenizer.from_pretrained(args.trans_ppl_model)

    ParallelCorpus.init()
    number_of_documents = 0

    with smart_open.open(args.input_path, "r", encoding="utf-8") as fp_in:
        with tqdm(total=args.first_n, desc="Documents indexed") as pbar:
            for batch_no, batch in enumerate(batched(fp_in, args.batch_size)):
                docs = []
                parsed = list(map(lambda x: x.split("\t", 1), batch))
                origs = list(map(lambda x: x[0], parsed))
                translations = list(map(lambda x: x[1], parsed))
                vectors = model.encode(origs)
                labse_orig_vectors = labse_model.encode(origs)
                labse_translations_vectors = labse_model.encode(translations)

                for orig, trans, vector, labse_orig_vector, labse_trans_vector in zip(
                    origs,
                    translations,
                    vectors,
                    labse_orig_vectors,
                    labse_translations_vectors,
                ):
                    if len(orig.strip()) <= args.filter_by_len:
                        number_of_documents += 1
                        pbar.update(1)
                        docs.append(
                            ParallelCorpus(
                                orig=orig,
                                trans=trans,
                                from_lang=args.from_lang,
                                to_lang=args.to_lang,
                                all_mpnet_base_v2_vector=list(vector),
                                labse_orig_vector=list(labse_orig_vector),
                                labse_trans_vector=list(labse_trans_vector),
                                orig_len=len(orig),
                                trans_len=len(trans),
                                labse_distance=float(
                                    sbert_utils.cos_sim(
                                        labse_orig_vector, labse_trans_vector
                                    )[0][0]
                                ),
                                orig_ppl=get_bpc(
                                    model=orig_ppl_model,
                                    tokenizer=orig_ppl_tokenizer,
                                    text=orig,
                                    device=args.device,
                                ),
                                trans_ppl=get_bpc(
                                    model=trans_ppl_model,
                                    tokenizer=trans_ppl_tokenizer,
                                    text=trans,
                                    device=args.device,
                                ),
                            ).to_dict(include_meta=True)
                        )

                print(f"Batch {batch_no}: Indexing another {len(docs)} documents")
                bulk_write(connections.get_connection(), docs)

                if number_of_documents >= args.first_n:
                    break

        # Leftovers
        bulk_write(connections.get_connection(), docs)


if __name__ == "__main__":
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    if torch.cuda.is_available():
        default_device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=pathlib.Path,
        help="Path to the file with the parallel corpus in tab separated format",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=1000,
        help="Number of sentences to draw from the input file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Size of the batch to embed and index",
    )
    parser.add_argument(
        "--filter-by-len",
        type=int,
        default=1000000,
        help="Do not include instructions longer than that",
    )
    parser.add_argument(
        "--transformer_model",
        type=str,
        help="Name of the sentence-transformer model to use",
        default="all-mpnet-base-v2",
    )
    parser.add_argument(
        "--labse_model",
        type=str,
        help="Name of the sentence-transformer model to use",
        default="LaBSE",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use for the models",
    )

    parser.add_argument(
        "--drop-index",
        default=False,
        help="Drop the index before indexing the corpus",
        action="store_true",
    )
    parser.add_argument(
        "--from-lang",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--to-lang",
        type=str,
        default="uk",
    )

    parser.add_argument(
        "--orig-ppl-model",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--trans-ppl-model",
        type=str,
        default="benjamin/gpt2-wechsel-ukrainian",
    )

    args = parser.parse_args()

    main(args=args)
