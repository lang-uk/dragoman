from datasets import load_dataset
from numpy.random import default_rng
import argparse

parser = argparse.ArgumentParser(description='Dataset generator for finetuning.')

parser.add_argument('--N', type=int, default=200_000,
                    help='Amount of samples to select from the dataset.')

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for the dataset shuffle')

parser.add_argument('--dataset', type=str, default="paracrawl.jsonlines",
                    help='Input file with the dataset.')


args = parser.parse_args()

N: int = args.N

data = load_dataset("json", data_files=args.dataset, split="train")

rng = default_rng(seed=args.seed)

full_dataset_size = len(data)

# generate datapoints' indices without repetitions
indices = rng.choice(full_dataset_size, size=N, replace=False)

data = data.select(indices)
print(len(data))
for i in range(5):
    idx = (len(data) // 5)
    start = idx * i
    end = idx * (i + 1)
    if i == 4:
        end = len(data)
    print(start, end)
    data.select(range(start, end)).to_json(f"shard_{N}_{i}.jsonlines", orient="records", lines=True, force_ascii=False)


