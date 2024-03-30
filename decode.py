import argparse
from itertools import islice
import json
from pathlib import Path

import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import BeamSearchDecoderOnlyOutput
from peft import PeftModel
import torch
sacrebleu = evaluate.load("sacrebleu")


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class Prompter:
    def __init__(self, prompt):
        if prompt == "gracious":
            self.generate_prompt = self.generate_gracious_prompt
            self.separator = "### Response:"
        elif prompt == "basic":
            self.generate_prompt = self.generate_basic_prompt
            self.separator = "[/INST]"
        else:
            raise ValueError(f"Unknown prompt style: {prompt}")

    def generate_gracious_prompt(self, instruction: str) -> str:
        return  f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Respond with Ukrainian translations of English input. ### Input: {instruction} ### Response: "

    def generate_basic_prompt(self, instruction: str) -> str:
        return f"[INST] {instruction} [/INST] "


class BatchTranslator:
    def __init__(self, *, decode_beams: int, decode_batch_size: int, model, tokenizer, prompter):
        self.decode_beams = decode_beams
        self.decode_batch_size = decode_batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter

    @classmethod
    def from_args(cls, args):
        return cls(
            decode_beams=args.decode_beams,
            decode_batch_size=args.decode_batch_size,
            model=cls.load_model(args),
            tokenizer=cls.load_tokenizer(cls.get_base_model(args)),
            prompter=Prompter(args.prompt),
        )

    @classmethod
    def register(cls, parser):
        parser.add_argument(
            "--model_name_or_path",
            default="google/gemma-2b",
            type=str,
            help="Base model name, HuggingFace or local path. Example options: mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mistral-7B-v0.1 huggyllama/llama-7b meta-llama/Llama-2-7b-hf upstage/SOLAR-10.7B-v1.0 Unbabel/TowerBase-7B-v0.1",
        )
        parser.add_argument(
            "--exp",
            default="exps/gemma2b-translate-uk-0.22.full-lora.4bit.diff-tokenizer.bigger-alpha.sophiag.1m_filtered",
            type=str,
            help="experiment directory: where to save the model",
        )
        parser.add_argument("--decode_subset", default=["dev", "devtest", "wmt22test"], type=str, help="Dataset decode: dev for FLORES dev, devtest for FLORES devtest, test for FLORES test, wmt22test for WMT22 test.")
        parser.add_argument("--decode_batch_size", default=1, type=int, help="Decoding batch size, should be small enough to accommodate all beams.")
        parser.add_argument("--decode_beams", default=10, type=int, help="Number of beams to use during decoding. Set to 0 to avoid decoding in the training loop.")
        parser.add_argument("--prompt", default="gracious", choices=["gracious", "basic"], type=str, help="Prompt style. Gracious uses a lot of words, basic uses [INST] [/INST].")

    @classmethod
    def make_result(cls):
        return {'id': [], 'rank': [], 'logprob': [], 'src': [], 'hyp': [], 'ref': [], 'bleu': []}

    @classmethod
    def get_base_model(self, args):
        path = Path(args.model_name_or_path)
        if path.is_dir():
            return json.loads((path / 'adapter_config.json').read_text())['base_model_name_or_path']
        else:
            return args.model_name_or_path

    @classmethod
    def load_model(self, args):
        model = AutoModelForCausalLM.from_pretrained(
            self.get_base_model(args),
            device_map="cuda",
            torch_dtype=torch.float16,
        )

        peft_model = PeftModel.from_pretrained(
            model,
            args.model_name_or_path,
            device_map="cuda",
        )
        peft_model = peft_model.merge_and_unload()
        peft_model = peft_model.half()
        return peft_model

    @classmethod
    def load_tokenizer(self, base_model, model_max_length=2048):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            model_max_length=model_max_length,
            use_fast=False,
            add_eos_token=False,
            add_bos_token=False,
            pad_token="<s>",
            padding_side="left",
        )
        return tokenizer

    @torch.inference_mode()
    def __call__(self, ids, sources, references):
        inputs = self.tokenizer(
            [self.prompter.generate_prompt(source) for source in sources],
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
                num_beams=self.decode_beams,
                num_return_sequences=self.decode_beams,
                #num_beam_groups=self.decode_beams//5,
            ),
        )

        logprobs = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices,
            normalize_logits=True
        ).sum(dim=-1)
        # output_length = np.sum(transition_scores.numpy() < 0, axis=1)

        strings = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        result = self.make_result()
        for example_id, src, ref, batch in zip(
            ids,
            sources,
            references,
            batched(zip(logprobs, strings), self.decode_beams)
        ):
            for rank, (logprob, output) in enumerate(batch):
                if self.prompter.separator in output:
                    _, output = output.split(self.prompter.separator, 1)
                    output = output.strip()
                else:
                    output = f"##ERROR: did not find separator {self.prompter.separator} in the output"
                print(example_id, logprob.item(), output)
                result['id'].append(example_id)
                result['rank'].append(rank)
                result['logprob'].append(logprob.item())
                result['src'].append(src)
                result['ref'].append(ref)
                result['hyp'].append(output)
                result['bleu'].append(sacrebleu.compute(predictions=[output], references=[ref])['score'])
        return result

    def report(self, output_path, dataset):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(output_path, force_ascii=False)

        # measure top-1 bleu
        dataset_top1 = dataset.filter(lambda x: x["rank"] == 0, load_from_cache_file=False)
        results = sacrebleu.compute(predictions=dataset_top1["hyp"], references=dataset_top1["ref"])
        output_path.with_suffix('.results').write_text(json.dumps(results, ensure_ascii=False))
        print(results)

        return results

    def decode_wmt22(self, exp: str):
        # https://github.com/huggingface/datasets/issues/4709
        dataset = load_dataset("text", data_files={
            "en": "data/wmt22/test.en-uk.en",
            "uk": "data/wmt22/test.en-uk.uk",
        })

        dataset = concatenate_datasets([dataset["en"].rename_column("text", "source"),
                                        dataset["uk"].rename_column("text", "target")], axis=1)
        dataset = dataset.add_column("id", list(range(len(dataset))))

        columns = ["id", "source", "target"]
        dataset = dataset.select_columns(columns)
        dataset = dataset.map(
            self,
            batched=True,
            batch_size=self.decode_batch_size,
            input_columns=columns,
            remove_columns=columns,
            load_from_cache_file=False,
        )

        return self.report(Path(exp) / f"beam{self.decode_beams}.wmt22test.jsonl", dataset)

    def decode_flores(self, exp: str, decode_subset: str, indices=None):
        dataset = load_dataset("facebook/flores", "eng_Latn-ukr_Cyrl", trust_remote_code=True)[decode_subset]
        if indices is not None:
            dataset = dataset.select(indices)
        columns = ["id", "sentence_eng_Latn", "sentence_ukr_Cyrl"]
        dataset = dataset.select_columns(columns)
        dataset = dataset.map(
            self,
            batched=True,
            batch_size=self.decode_batch_size,
            input_columns=columns,
            remove_columns=columns,
            load_from_cache_file=False,
        )

        return self.report(Path(exp) / f"beam{self.decode_beams}.{decode_subset}.jsonl", dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BatchTranslator.register(parser)
    args = parser.parse_args()

    translator = BatchTranslator.from_args(args)
    if "wmt22test" == args.decode_subset:
        translator.decode_wmt22(exp=args.exp)
    else:
        translator.decode_flores(exp=args.exp, decode_subset=args.decode_subset)
