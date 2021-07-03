import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import read_in_merged_data, extract_label

# Use LaTEX font
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})


def print_summary_statistics(df_merged):
    print(f"Number of samples: {len(df_merged)}")
    print(f"Number of classes: {df_merged['label'].nunique()}")
    print(
        f"Median number of words per sample: {round(np.median([len(transcript.split()) for transcript in list(df_merged['transcript'])]))}")
    print(
        f"Min number of words per sample: {np.min([len(transcript.split()) for transcript in list(df_merged['transcript'])])}")
    print(
        f"Max number of words per sample: {np.max([len(transcript.split()) for transcript in list(df_merged['transcript'])])}")


def plot_sample_length_distribution(df):
    """Plots a histogram with the distribution of the transcript lengths stored in a data frame.

    # Parameters:
        df: pd.DataFrame, a data frame containing a "transcript" column

    # Raises:
        AssertionError : if "transcript" is not a column in `df` 
    """
    assert "transcript" in df.columns

    plt.hist([len(transcript.split())
              for transcript in list(df['transcript'])], 50)
    plt.ylabel("Number of samples")
    plt.xlabel("Approximate number of words per sample")
    plt.title("Distribution of transcript lengths in the TED dataset")

    plt.show()


def plot_y_distribution(df, percentages=False):
    """Plots a horizontal bar graph with the distribution of the labels in a data frame.

    # Parameters:
        df: pd.DataFrame, a data frame containing a "label" column
        percentages: bool, whether to plot the proprtion of observations a given label
                            corresponds to, or just the raw counts

    # Raises:
        AssertionError : if "label" is not a column in `df` 
    """
    assert "label" in df.columns

    if percentages:
        (df["label"].value_counts()/len(df)*100.0).sort_values().plot.barh()
        plt.xlabel("Percentage of observations (%)")
    else:
        df["label"].value_counts().sort_values().plot.barh()
        plt.xlabel("Number of observations")

    plt.ylabel("Label")
    plt.title("Distribution of labels in the TED dataset")

    plt.show()


def main():
    # Merge `main.csv` and `transcripts.csv`
    df_merged = read_in_merged_data(path_to_tedtalks='')

    # Extract TED label from the metadata
    df_merged["label"] = df_merged["tags"].apply(extract_label)

    # Conduct EDA
    print_summary_statistics(df_merged)
    plot_sample_length_distribution(df_merged)
    plot_y_distribution(df_merged, True)


if __name__ == "__main__":
    main()
