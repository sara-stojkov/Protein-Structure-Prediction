import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_IDX = len(AMINO_ACIDS)  # 20 - "unknown" amino acid index
N_AA_CLASSES = len(AMINO_ACIDS) + 1  # 21

SS3_TO_IDX = {"H": 0, "E": 1, "C": 2}


def load_csv_dataset(path):
    df = pd.read_csv(path)
    df = df[["seq", "sst3", "len_x"]].copy()
    df = df.rename(columns={"len_x": "len"})
    df = df.dropna(subset=["seq", "sst3"])
    return df


def one_hot_encode_sequence(seq, max_len):
    arr = np.zeros((max_len, N_AA_CLASSES), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        idx = AA_TO_IDX.get(aa, UNK_IDX)
        arr[i, idx] = 1.0
    return arr


def encode_labels(sst3_string, max_len):
    """sst3_string: for example 'HHHEEECCC' -> np.array shape (max_len,), -1 for padding"""
    labels = np.full((max_len,), fill_value=-1, dtype=np.int64)
    for i, ch in enumerate(sst3_string[:max_len]):
        labels[i] = SS3_TO_IDX.get(ch, -1)
    return labels


def build_mask(seq_len_actual, max_len):
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[:min(seq_len_actual, max_len)] = 1.0
    return mask


def dataframe_to_tensors(df, max_len):
    """returns X (n, max_len, 21), y (n, max_len), mask (n, max_len)"""
    n = len(df)
    X = np.zeros((n, max_len, N_AA_CLASSES), dtype=np.float32)
    y = np.full((n, max_len), fill_value=-1, dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.float32)

    for i, row in enumerate(df.itertuples()):
        X[i] = one_hot_encode_sequence(row.seq, max_len)
        y[i] = encode_labels(row.sst3, max_len)
        mask[i] = build_mask(len(row.seq), max_len)

    return X, y, mask