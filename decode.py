import argparse
from itertools import islice
import json
import logging
from pathlib import Path

import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import BeamSearchDecoderOnlyOutput
from peft import PeftModel

sacrebleu = evaluate.load("sacrebleu")
logger = logging.getLogger(__name__)


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class BatchTranslator:
    def __init__(self, args):
        self.model = self.load_model(args, args.checkpoint)
        self.tokenizer = self.load_tokenizer(args)
        self.beams = args.beams
        self.batch_size = args.batch_size

    @classmethod
    def register(cls, parser):    
        parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1", type=str)
        parser.add_argument("--checkpoint", help="path to adapter checkpoint", required=True)
        parser.add_argument("--output_dir", default="eval-beams", type=Path)
        parser.add_argument("--subset", default="devtest", type=str)
        parser.add_argument("--batch_size", default=2, type=int, help="should be small enough to accommodate all beams")
        parser.add_argument("--beams", default=10, type=int)

    @classmethod
    def make_result(cls):
        return {'id': [], 'rank': [], 'logprob': [], 'src': [], 'hyp': [], 'ref': [], 'bleu': []}

    def load_model(self, args, checkpoint):
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="cpu",
        )

        peft_model = PeftModel.from_pretrained(
            model,
            checkpoint,
            device_map="cpu",
        )
        peft_model = peft_model.merge_and_unload()
        peft_model = peft_model.half().cuda()
        return peft_model

    def load_tokenizer(self, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            model_max_length=1024,
            use_fast=False,
            add_eos_token=False,
            add_bos_token=False,
            pad_token="<s>",
            padding_side="left",
        )
        return tokenizer

    def generate_prompt(self, instructions: list[str]) -> list[str]:
        return [f"[INST] {instruction} [/INST]" for instruction in instructions]

    def __call__(self, ids, sources, references):
        inputs = self.tokenizer(
            self.generate_prompt(sources),
            return_tensors="pt",
            padding=True
        )
        outputs: BeamSearchDecoderOnlyOutput = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            use_cache=True,
            generation_config=GenerationConfig(
                pad_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=self.beams,
                num_return_sequences=self.beams
            ),
        )

        logprobs = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices,
            normalize_logits=True
        ).sum(dim=-1)
        #output_length = np.sum(transition_scores.numpy() < 0, axis=1)

        strings = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        result = self.make_result()
        for example_id, src, ref, batch in zip(
            ids,
            sources,
            references,
            batched(zip(logprobs, strings), self.beams)
        ):
            for rank, (logprob, output) in enumerate(batch):
                if "[/INST]" in output:
                    _, output = output.split("[/INST]", 1)
                    output = output.strip()
                else:
                    output = "##ERROR"
                print(example_id, logprob.item(), output)
                result['id'].append(example_id)
                result['rank'].append(rank)
                result['logprob'].append(logprob.item())
                result['src'].append(src)
                result['ref'].append(ref)
                result['hyp'].append(output)
                result['bleu'].append(sacrebleu.compute(predictions=[output], references=[ref])['score'])
        return result


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    BatchTranslator.register(parser)
    args = parser.parse_args()

    logger.info(f"Loading checkpoint {args.checkpoint}")
    checkpoint_slug = args.checkpoint.replace("/", "-")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{checkpoint_slug}.beam{args.beams}.{args.subset}.jsonl"

    translator = BatchTranslator(args)
    dataset = load_dataset("facebook/flores", "eng_Latn-ukr_Cyrl", trust_remote_code=True)[args.subset]
    #dataset = dataset.select(range(10)) # for testing

    columns = ["id", "sentence_eng_Latn", "sentence_ukr_Cyrl"]
    dataset = dataset.select_columns(columns)
    dataset = dataset.map(
        translator,
        batched=True,
        batch_size=args.batch_size,
        input_columns=columns,
        remove_columns=columns,
        load_from_cache_file=False,
    )
    dataset.to_json(output_path, force_ascii=False)

    # measure top-1 bleu
    dataset = dataset.filter(lambda x: x["rank"] == 0, load_from_cache_file=False)
    results = sacrebleu.compute(predictions=dataset["hyp"], references=dataset["ref"])
    output_path.with_suffix('.results').write_text(json.dumps(results, ensure_ascii=False))
    print(results)
