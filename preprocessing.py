import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def read_in_merged_data(path_to_tedtalks=''):
    """ Merge transcripts.csv with ted_main.csv

    # Arguments:
        path_to_data : str
            Specify the path to the `tedtalks` folder containing the data

    # Returns:
        df_merged : pd.DataFrame
            A data frame containing the merged transcripts.csv and ted_main.csv files
    """

    # Read in the data
    df_main = pd.read_csv(path_to_tedtalks + 'tedtalks/ted_main.csv')
    df_transcripts = pd.read_csv(path_to_tedtalks + 'tedtalks/transcripts.csv')

    # Merge data frames on `url`, since some ted talks in `ted_main.csv` do not have transcripts in `transcripts.csv`
    df_merged = df_main.merge(df_transcripts, how="inner", on="url")

    return df_merged


def extract_label(tags):
    """Returns the T/E/D label for a TED talk based on its keyword tags,
       where T = Technology, E = Entertainment, D = Design.

    # Parameters:
        tags: str, containing a list of keyword tags for a TED Talk.
                Example input: "['computers', 'design', 'technology']"

    # Returns:
        label: str, T/E/D label
                Options: "Other", "T", "E", "D", "TE", "TD", "ED" and "TED"
    """

    # Cast tags from string to a list e.g. "['technology', 'design']" to ['technology', 'design']
    tags = eval(tags)

    # Cast all tags to lower case
    tags = [tag.lower() for tag in tags]

    # Extract which T/E/D categories are contained in the keyword tags
    label_categories = ['technology', 'entertainment', 'design']
    ted_category = set(label_categories).intersection(tags)

    # Sort the extracted T/E/D categories in T-E-D order i.e. descending order
    ted_category = sorted(list(ted_category), reverse=True)

    # Extract and capitalize the first letter of each T/E/D component
    ted_category = [c[0].upper() for c in ted_category]

    # Cast the list to a string e.g. ["T", "D"] to "TD"
    ted_category = "".join(ted_category)

    # If no T/E/D category, label the talk as "other"
    label = "Other" if ted_category == "" else ted_category

    return label


def train_val_test_split(df, random_state=42, validation_set_size=0.2, test_set_size=0.2):
    """Splits a data frame into train, validation and test sets – stratified according to
      label proportions.

    # Parameters:
        df: pd.DataFrame, a data frame containing a "transcript" and "label" column
        random_state: int, for reproducibility
        validation_set_size: int, the absolute size of the validation set to create from `df`
                              Default: 0.2 – 20% of the 2467 observations
        test_set_size: int, the absolute size of the test set to create from `df`
                              Default: 0.2 - 20% of the 2467 observations

    # Returns:
        df_train, df_val, df_test: pd.DataFrame
            `df` split into train, validation and test sets with "transcript"
            and "label" columns.

    # Raises:
        AssertionError : if either "transcript" or "label" are not columns in `df` 
    """
    assert "transcript" in df.columns and "label" in df.columns

    # Keep only the "transcript" and "label" columns in `df_all`
    df_all = pd.DataFrame(df[["transcript", "label"]])

    # Extract 20% of observations randomly for the validation set,
    # but stratify on the "label" column to preserve label proportions
    df_train, df_val = train_test_split(df_all,
                                        test_size=validation_set_size,
                                        random_state=random_state,
                                        stratify=df_all["label"])

    # Extract 20% of observations randomly for the test set,
    # but stratify on the "label" column to preserve label proportions
    df_train, df_test = train_test_split(df_train,
                                         test_size=test_set_size /
                                         (1 - validation_set_size),
                                         random_state=random_state,
                                         stratify=df_train["label"])

    return df_train, df_val, df_test


def encode_labels(df_train, df_val, df_test):
    """ Encodes string labels in data frames into integers.

    # Parameters:
        df_train, df_val, df_test: pd.DataFrame
            Data frames containing a "label" column to encode, 
            where the "label" column consists of strings.

    # Returns:
        y_train, y_val, y_test: np.ndarray of shape (n_obs,)
            Integer encoded response for training, validation, and testing.
        label_tokenizer: tf.keras.preprocessing.text.Tokenizer
            Used later to map integer encodings back to the string labels.

    # Raises:
        AssertionError : if "label" is not a column in `df_train`,
                          `df_val` or `df_test`
    """
    assert all("label" in df.columns for df in [df_train, df_val, df_test])

    # Fit the label tokenizer on the training data
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(df_train['label'])

    # Convert from strings ("Other","T", "E", ... "TED") to integers in the range [1, 8]
    y_train = np.array(label_tokenizer.texts_to_sequences(df_train['label']))
    y_val = np.array(label_tokenizer.texts_to_sequences(df_val['label']))
    y_test = np.array(label_tokenizer.texts_to_sequences(df_test['label']))

    # Subtract 1 to cast encoded labels from range [1, 8] to range [0, 7].
    y_train, y_val, y_test = y_train - 1, y_val - 1, y_test - 1

    # Flatten to np.ndarray of shape (n_obs,)
    y_train, y_val, y_test = y_train.flatten(), y_val.flatten(), y_test.flatten()

    return y_train, y_val, y_test, label_tokenizer


def vectorize_transcripts(df_train, df_val, df_test, y_train, top_k_features=20000):
    """Tokenize and then vectorize the transcripts using the TF-IDF approach.

    # Parameters:
      df_train, df_val, df_test: pd.DataFrame
          Data frames containing a "transcript" column to encode, 
          where the "transcript" column consists of text strings.
      top_k_features: int
          The number of n-grams to keep for the bag-of-words.
          Only the `top_k_features` most important features will
          be retained.
      y_train: np.ndarray of shape (n_train,)
          Integer encoded labels for the training data, used
          for feature importance for choosing which tokens to retain.

    # Returns:
      X_train, X_val, X_test: scipy.sparse.csr.csr_matrix of shape (n_obs, top_k_features)
          Sparse matrices (to store efficiently) with the TF-IDF vectorized transcripts.
      transcript_vectorizer: sklearn.feature_extraction.text import TfidfVectorizer
          Based on training data, so can be used later to vectorize new data
      ngram_selector: sklearn.feature_selection.SelectKBest  
          Based on training data, so can be used later to select 
          the same 20000 best n-grams for new data  

    # Raises:
      AssertionError : if "transcript" is not a column in `df_train`,
                        `df_val` or `df_test`
    """
    assert all("transcript" in df.columns for df in [
               df_train, df_val, df_test])

    # Construct the TF-IDF vectorizer, considering both 1-grams and 2-grams as tokens
    transcript_vectorizer = TfidfVectorizer(ngram_range=(
        1, 2), strip_accents='unicode', min_df=2)

    # Learn vocabulary from training transcripts and vectorize training transcripts
    X_train = transcript_vectorizer.fit_transform(df_train["transcript"])

    # Vectorize validation and test transcripts
    X_val = transcript_vectorizer.transform(df_val["transcript"])
    X_test = transcript_vectorizer.transform(df_test["transcript"])

    # Select the 'top_k_features' most important n-grams in predicting the response,
    # using the chi-squared score as a measure of association between each
    # feature and the response in the training data.
    ngram_selector = SelectKBest(chi2, k=min(top_k_features, X_train.shape[1]))
    ngram_selector.fit(X_train, y_train)

    # Retain only the 'top_k_features' for each of the X matrices
    X_train = ngram_selector.transform(X_train).astype('float32')
    X_val = ngram_selector.transform(X_val).astype('float32')
    X_test = ngram_selector.transform(X_test).astype('float32')

    return X_train, X_val, X_test, transcript_vectorizer, ngram_selector
