import json
import pandas as pd

# TODO: clean up this file and make it more flexible

df = pd.read_csv("../data/parallel_corpus_plus_hash.csv.bz2")

df["abs_len_difference"] = abs(df["orig_len"] - df["trans_len"])
df["sum_ppl"] = df["orig_ppl"] + df["trans_ppl"]

df_filtered = df[
    (df["detected_to_lang"] == "uk-True")
    & (df["detected_from_lang"] == "en-True")
    & (df["labse_distance"] > 0.91)
    & (df["abs_len_difference"] < 50)
    & (df["sum_ppl"] < 3.33)
    & (df["orig_len"] <= 1024)
]

args_template = "[INST] {orig} [/INST] {translated}"
args_filter_by_len = 1024

with open("/tmp/paracrawl_filtered.jsonlines", "w") as fp_out:
    for index, row in df_filtered.iterrows():
        sample = row.to_dict()
        text = args_template.format(
            orig=sample["orig"].strip(), translated=sample["trans"].strip()
        )
        if len(text) <= args_filter_by_len:
            fp_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
