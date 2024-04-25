# Dragoman: SOTA English to Ukrainian Machine Translation

[![Open In HF🤗 Space ](https://img.shields.io/badge/Open%20Demo-%F0%9F%A4%97%20Space-yellow)](https://huggingface.co/spaces/lang-uk/dragoman)
[![Download model 🤗 ](https://img.shields.io/badge/Download%20Model-%F0%9F%A4%97%20Model-yellow)](https://huggingface.co/lang-uk/dragoman)

This repository is an official implementation of paper [Setting up the Data Printer with Improved English to Ukrainian Machine Translation](https://arxiv.org/abs/2404.15196) (accepted to UNLP 2024 at LREC-Coling 2024).
By using a two-phase data cleaning and data selection approach we have achieved SOTA performance on FLORES-101 English-Ukrainian devtest subset with **BLEU** `32.34`.


#### Online demo: [https://huggingface.co/spaces/lang-uk/dragoman](https://huggingface.co/spaces/lang-uk/dragoman)


## How to use

We designed this model for sentence-level English -> Ukrainian translation.
Performance on multi-sentence texts is not guaranteed, please be aware.


#### Running the model


```python
# pip install bitsandbytes transformers peft torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

config = PeftConfig.from_pretrained("lang-uk/dragoman")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=float16,
    bnb_4bit_use_double_quant=False,
)

model = MistralForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config
)
model = PeftModel.from_pretrained(model, "lang-uk/dragoman").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", use_fast=False, add_bos_token=False
)

input_text = "[INST] who holds this neighborhood? [/INST]" # model input should adhere to this format
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

### Running the model with mlx-lm on an Apple computer


We merged Dragoman PT adapter into the base model and uploaded the quantized version of the model into https://huggingface.co/lang-uk/dragoman-4bit.

You can run the model using [mlx-lm](https://pypi.org/project/mlx-lm/).


```
python -m mlx_lm.generate --model lang-uk/dragoman-4bit --prompt '[INST] who holds this neighborhood? [/INST]' --temp 0 --max-tokens 100
```

MLX is a recommended way of using the language model on an Apple computer with an M1 chip and newer.


### Running the model with llama.cpp

We converted Dragoman PT adapter into the [GGLA format](https://huggingface.co/lang-uk/dragoman/blob/main/ggml-adapter-model.bin).

You can download the [Mistral-7B-v0.1 base model in the GGUF format](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) (e.g. mistral-7b-v0.1.Q4_K_M.gguf)
and use `ggml-adapter-model.bin` from this repository like this:

```
./main -ngl 32 -m mistral-7b-v0.1.Q4_K_M.gguf --color -c 4096 --temp 0 --repeat_penalty 1.1 -n -1 -p "[INST] who holds this neighborhood? [/INST]" --lora ./ggml-adapter-model.bin
```

### Benchmark Results against other models on FLORES-101 devset


| **Model**                                   | **BLEU** $\uparrow$ | **spBLEU** | **chrF** | **chrF++** |
|---------------------------------------------|---------------------|-------------|----------|------------|
| **Finetuned**                               |                     |             |          |            |
| Dragoman P, 10 beams                        | 30.38               | 37.93       | 59.49    | 56.41      |
| Dragoman PT, 10 beams                       | **32.34**           | **39.93**   | **60.72**| **57.82**  |
|---------------------------------------------|---------------------|-------------|----------|------------|
| **Zero shot and few shot**                  |                     |             |          |            |
| LLaMa-2-7B 2-shot                           | 20.1                | 26.78       | 49.22    | 46.29      |
| RWKV-5-World-7B 0-shot                      | 21.06               | 26.20       | 49.46    | 46.46      |
| gpt-4 10-shot                               | 29.48               | 37.94       | 58.37    | 55.38      |
| gpt-4-turbo-preview 0-shot                  | 30.36               | 36.75       | 59.18    | 56.19      |
| Google Translate 0-shot                     | 25.85               | 32.49       | 55.88    | 52.48      |
|---------------------------------------------|---------------------|-------------|----------|------------|
| **Pretrained**                              |                     |             |          |            |
| NLLB 3B, 10 beams                           | 30.46               | 37.22       | 58.11    | 55.32      |
| OPUS-MT, 10 beams                           | 32.2                | 39.76       | 60.23    | 57.38      |


### Training Dataset and Resources

#### Resulting Datasets
Cleaned Paracrawl (first phase): [lang-uk/paracrawl_3m](https://huggingface.co/datasets/lang-uk/paracrawl_3m)  
Cleaned Multi30K (second phase): [lang-uk/multi30k-extended-17k](https://huggingface.co/datasets/lang-uk/multi30k-extended-17k)


#### Training steps

For more details, please refer to the paper.

1. First phase: Data Cleaning of [Paracrawl](https://huggingface.co/datasets/Helsinki-NLP/opus_paracrawl) dataset.
2. Second phase: Unsupervised Data Selection using k-fold perplexity filtering on [Extended Multi30k-Uk](https://huggingface.co/datasets/turuta/Multi30k-uk).
```bash
# export turuta/Multi30k-uk to a local dataset
python download_dataset.py

# generate k-folds for perplexity evaluation
python generate_dataset.py --N 29_000 --dataset multi-30k-uk.jsonl

# train 5 models on 5 folds, resume from previous phase
python finetune_ppl.py --N 29_000 --run_type folds --prefix fold-training --lora_checkpoint exps/dragoman-p --lr 2e-5

# calculate perplexity for OOB data
python perplexity_evaluate.py --N 29_000 --lr 2e-5 --prefix fold-training

# apply perplexity filtering
python ppl_analysis.py --threshold 60

# train on the selected data
python finetune_ppl.py --N 29_000 --run_type cleaned --prefix fold-training --lora_checkpoint exps/dragoman-p --lr 2e-5

# train on the full data for compatison
python finetune_ppl.py --N 29_000 --run_type full --prefix fold-training --lora_checkpoint exps/dragoman-p --lr 2e-5

# evaluate on flores dev
python decode.py --checkpoint exps/fold-training_epochs_1_lr_2e-05_R_128_ALPH_256_N_29000_full --subset dev

# evaluate on flores devtest
python decode.py --checkpoint exps/fold-training_epochs_1_lr_2e-05_R_128_ALPH_256_N_29000_full --subset devtest

```

## Citation

TBD (accepted to UNLP 2024 at LREC-Coling 2024)


## Authors

Yurii Paniv, Dmytro Chaplynskyi, Nikita Trynus, Volodymyr Kyrylov 