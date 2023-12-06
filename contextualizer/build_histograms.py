import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return a DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path, compression="bz2")
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame by adding new columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Add new columns
    df["abs_len_difference"] = abs(df["orig_len"] - df["trans_len"])
    df["sum_ppl"] = df["orig_ppl"] + df["trans_ppl"]

    return df


def plot_histograms(df: pd.DataFrame, output_dir: str):
    """
    Create and save histograms for selected columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        output_dir (str): Directory to save the histograms.
    """
    columns_to_plot = [
        "orig_len",
        "trans_len",
        "labse_distance",
        "orig_ppl",
        "trans_ppl",
        "abs_len_difference",
        "sum_ppl",
        "detected_from_lang",
        "detected_to_lang",
    ]

    for column in tqdm(columns_to_plot):
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=50, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{output_dir}/{column}_histogram.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process and visualize a large dataset."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the histograms."
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = load_data(args.input_file)

    # Process data
    print("Processing data...")
    processed_data = process_data(data)

    # Plot and save histograms
    print("Plotting histograms...")
    plot_histograms(processed_data, args.output_dir)

    print("Script completed successfully.")


if __name__ == "__main__":
    main()
