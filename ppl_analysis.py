# %%
import pandas as pd
import argparse
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

N = 29_000  # args.N


parser = argparse.ArgumentParser(description="Dataset generator for finetuning.")


# Required positional argument
parser.add_argument(
    "--threshold", type=int, default=60, help="Threshold for filtering."
) 

args = parser.parse_args()

threshold = args.threshold

print("Selected threshold:", threshold, "percentile.")

# %%
output_df = pd.DataFrame()
faulty_df = pd.DataFrame()
total_df = pd.DataFrame()

#thres_colors = ["red", "green", "blue", "purple", "cyan"]
#plt.title("Distribution of sentence pair log probabilities")
#plt.rcParams['font.size'] = 14
# remove spines from plots

#plt.rcParams['font.size'] = 14


for i in range(0, 5):
    print(f"Processing shard {i}")
    df = pd.read_csv(f"shard_{N}_{i}_ppl.csv", header=None)
    df = df.join(pd.read_json(f"shard_{N}_{i}.jsonlines", lines=True), how="inner")

    df = df.copy()

    df["log"] = df[0].apply(lambda x: np.log(x))
    filter_column = "log"
    mean = df[filter_column].mean()
    std = df[filter_column].std()

    if threshold == -1:
        upper = mean + 2 * std
    else:
        upper = np.percentile(df[filter_column], threshold)

    lower = 0
    df["out_of"] = df[filter_column].apply(lambda x: x > upper or x < lower)
    sns.histplot(data=df, x=filter_column, bins=100, label=f"Scores for fold {i + 1}")
    
    #plt.axvline(upper, color=thres_colors[i], label=f"60th percentile for fold {i + 1}", linestyle="dotted")


    drop_condtion = (df["out_of"] == False) & (df[0].isna() == False)
    output_df = pd.concat([output_df, df[drop_condtion].drop(columns=["out_of"])])
    faulty_df = pd.concat([faulty_df, df[~drop_condtion].drop(columns=["out_of"])])
    total_df = pd.concat([total_df, df])

#ax = plt.gca()
#ax.legend_ = None

#plt.gca().spines['top'].set_visible(False)
#plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#plt.xlabel("Negative log probability")
#plt.ylabel("Number of samples")
#plt.savefig("Log ppl distribution.svg", bbox_inches="tight")


output_df = output_df.reset_index(drop=True)
faulty_df = faulty_df.reset_index(drop=True)
total_df = total_df.reset_index(drop=True)


output_df.to_json(
    f"shard_{N}_ppl_filtered.jsonlines",
    index=False,
    force_ascii=False,
    orient="records",
    lines=True,
)

faulty_df.sort_values(0, ascending=False).to_json(
    f"shard_{N}_ppl_faulty.jsonlines",
    index=False,
    force_ascii=False,
    orient="records",
    lines=True,
)
print(f"Faulty: {len(faulty_df)}")
print(f"Filtered: {len(output_df)}")

