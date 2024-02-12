import argparse
import math
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
)
import time
from peft import prepare_model_for_kbit_training, PeftModel

from finetune import tokenize
from optimizers.sophia import SophiaG


parser = argparse.ArgumentParser("superalignment", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1", type=str)
parser.add_argument("--init", help="path to initial adapter checkpoint", default="exps/mistral-translate-uk-0.15.full-lora.4bit.diff-tokenizer.sophiag.3m_sorted_dataset", type=str)
parser.add_argument("--exp", type=str, required=True, help="path output experiment checkpoint")
parser.add_argument("--data", type=str, default=f"eval-beams/exps-mistral-translate-uk-0.15.full-lora.4bit.diff-tokenizer.sophiag.3m_sorted_dataset.beam25.jsonl", help="path to eval-beams jsonl file")
parser.add_argument("--neg", default=[1,5,10], type=int, nargs='+', help="indices of negative examples per positive example")
parser.add_argument("--lr", default=1e-7, type=float, help="learning rate")
parser.add_argument("--epochs", default=2, type=int, help="number of epochs")
parser.add_argument("--clip", default=0.1, type=float, help="max gradient norm fors clipping")
parser.add_argument("--warmup", default=100, type=int, help="number of warmup steps")
args = parser.parse_args()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    model_max_length=1024,
    use_fast=False,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=False,
)
tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorForTokenClassification(
    tokenizer,
    pad_to_multiple_of=1,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quant_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

model = PeftModel.from_pretrained(
    model,
    args.init,
)
model.config.use_cache = False

beam_data = load_dataset(
    "json",
    data_files=args.data,
    split="train"
)
dataset = Dataset.from_list(beam_data.to_pandas().groupby("id").apply(lambda x: {
    "id": x.iloc[0]["id"],
    "src": x.iloc[0]["src"],
    "ref": x.iloc[0]["ref"],
    "hypotheses": x["hyp"].tolist(),
    "ranks": x["rank"].tolist()
}).tolist())

dataset = dataset.map(lambda x: {
    'pos': tokenize(tokenizer, f'[INST] {x["src"]} [/INST] {x["ref"]}'),
    'neg': [tokenize(tokenizer, f'[INST] {x["src"]} [/INST] {hypo}') for hypo in [x["hypotheses"][i] for i in args.neg]]
})


def mce_forward(model, batch):
    """Minimum classification error forward pass"""

    pos = collator(batch['pos'])
    pos_forward = model(
        input_ids=pos['input_ids'],
        attention_mask=pos['attention_mask'],
        labels=pos['labels'],
    )
    neg_forwards = []
    for neg in batch['neg']:
        neg = collator(neg)
        neg_forwards.append(model(
            input_ids=neg['input_ids'],
            attention_mask=neg['attention_mask'],
            labels=neg['labels'],
        ).loss)

    neg_loss = torch.stack(neg_forwards).logsumexp(dim=-1) + math.log(1/len(neg_forwards))
    loss = pos_forward.loss - neg_loss
    return loss


def train(model, dataset, optimizer, args, cooldown=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 1
    steps = len(dataset) * args.epochs
    print(f'MCE training for {args.epochs} epochs')

    now = time.monotonic()
    for epoch in range(args.epochs):
        for batch in dataset.shuffle().iter(batch_size=8):
            loss = mce_forward(model, batch)

            if loss < 0:
                print(f'negative loss for examples {batch["id"]}: {loss.item()}, skipping batch', flush=True)
                print(batch['src'])
                print(batch['ref'])
                print([[x[i] for i in args.neg] for x in batch['hypotheses']])
                continue

            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip)

            if step < args.warmup:
                # linear warmup
                current_lr = (step/args.warmup) * args.lr
            elif cooldown:
                # linear cooldown
                current_lr = (1 - (step-args.warmup)/(steps-args.warmup)) * args.lr
            else:
                current_lr = args.lr
            optimizer.param_groups[0]['lr'] = current_lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step == 1 or step % 2 == 0:
                then = time.monotonic()
                print(f'{step:6} steps, {loss:.4f} loss,',
                    f'{current_lr:.8f} lr,', f'{grad_norm:.4f} grad norm, {then-now:.4f} elapsed', flush=True)
                now = then

            step += 1


def mark_lora_as_trainable_(model):
    assert model.peft_config['default'].bias == 'none'
    for n,p in model.named_parameters():
        if model.prefix in n:
            p.requires_grad_(True)

            
if __name__ == '__main__':
    mark_lora_as_trainable_(model)
    optimizer = SophiaG(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    train(model, dataset, optimizer, args)
    model.save_pretrained(args.exp)
