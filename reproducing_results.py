from preprocessing import read_in_merged_data, extract_label, train_val_test_split, encode_labels, vectorize_transcripts
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

###########################
## Conduct preprocessing ##
###########################

print("Conducting preprocessing... (takes approximately 20 seconds)")

# Merge `main.csv` and `transcripts.csv`
df_merged = read_in_merged_data(path_to_data='')

# Extract TED label from the metadata
df_merged["label"] = df_merged["tags"].apply(extract_label)

# Apply train-val-test split
df_train, df_val, df_test = train_val_test_split(df_merged)

# Encode labels as integers
y_train, y_val, y_test, label_tokenizer = encode_labels(
    df_train, df_val, df_test)

# Tokenize / vectorize transcripts using TF-IDF bag-of-words approach
X_train, X_val, X_test = vectorize_transcripts(
    df_train, df_val, df_test, y_train)

###########################
#### Produce results    ###
###########################

print("Producing final MLP results... ")

# Load the best MLP model (with the highest validation F1-score)
mlp_final_model = tf.keras.models.load_model("best_model")

# Predict using this model on the test data
preds_test = np.argmax(mlp_final_model.predict(X_test), axis=1)

# Compute and report the test accuracy and test F1 score
test_accuracy = accuracy_score(preds_test, y_test)
test_f1 = f1_score(y_test, preds_test, average='macro')
print("-"*50)
print(f"Final MLP test accuracy: {test_accuracy*100:.2f}%")
print(f"Final MLP F1-score: {test_f1*100:.2f}%")
print("-"*50)
