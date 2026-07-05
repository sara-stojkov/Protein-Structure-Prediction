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

def load_cb513(path, max_len=400):
    """
    parses CB513.npy (format Zhou & Troyanskaya: 700 pozicija x 57 vrednosti po proteinu)
    returns X (n, max_len, 21), y (n, max_len), mask (n, max_len)
    """
    raw = np.load(path)

    if raw.ndim == 2:
        raw = raw.reshape(-1, 700, 57)
    elif raw.ndim != 3:
        raise ValueError(f"Neočekivan oblik CB513 fajla: {raw.shape}")

    n_proteins, seq_len_raw, n_features = raw.shape
    print(f"CB513 učitan: {n_proteins} proteina, {seq_len_raw} pozicija, {n_features} vrednosti po poziciji")

    # SST-8 sequence in this format (standard Zhou&Troyanskaya): L B E G I H S T + "no seq"
    SST8_ORDER = ["L", "B", "E", "G", "I", "H", "S", "T"]
    # Mapping 8->3 classes (same logic as with csv files): H/G/I -> H, E/B -> E, L/S/T -> C
    SST8_TO_3 = {
        "H": "H", "G": "H", "I": "H",
        "E": "E", "B": "E",
        "L": "C", "S": "C", "T": "C",
    }

    X = np.zeros((n_proteins, max_len, N_AA_CLASSES), dtype=np.float32)
    y = np.full((n_proteins, max_len), fill_value=-1, dtype=np.int64)
    mask = np.zeros((n_proteins, max_len), dtype=np.float32)

    for i in range(n_proteins):
        aa_onehot = raw[i, :, 0:21]       # (700, 21)
        ss_onehot = raw[i, :, 22:30]      # (700, 8)

        # "actual" position = has exactly one 1 in aa_onehot (not all zeros -> padding/no-seq)
        valid_positions = aa_onehot.sum(axis=1) > 0
        actual_len = int(valid_positions.sum())

        use_len = min(actual_len, max_len)

        X[i, :use_len, :21] = aa_onehot[:use_len]
        mask[i, :use_len] = 1.0

        ss_idx = ss_onehot[:use_len].argmax(axis=1)  # indeks SST-8 klase po poziciji
        for pos in range(use_len):
            sst8_letter = SST8_ORDER[ss_idx[pos]]
            sst3_letter = SST8_TO_3[sst8_letter]
            y[i, pos] = SS3_TO_IDX[sst3_letter]

    return X, y, mask