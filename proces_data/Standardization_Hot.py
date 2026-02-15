import numpy as np
from sklearn.preprocessing import OneHotEncoder

# =========================
# Path X (Input e Output)
# =========================
X_train_aug_path = "....../paderborn_university/dataset/5012/X_train_raw_5012_aug.npy"
X_train_std_path = "....../paderborn_university/dataset/5012/X_train_raw_5012_noaug.npy"
X_test_std_path = "......./paderborn_university/dataset/5012/X_test_raw_5012_noaug.npy"

Y_train_aug_path = "......./paderborn_university/dataset/5012/Y_train_raw_5012_aug.npy"
Y_train_std_path = "......./paderborn_university/dataset/5012/Y_train_raw_5012_noaug.npy"
Y_test_std_path = "......../paderborn_university/dataset/5012/Y_test_raw_5012_noaug.npy"


def normalize_per_signal(X):
    """
    X shape: (n_samples, signal_length)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)


# =========================
# Load X
# =========================
X_train_std = np.load(X_train_std_path).astype(np.float32)
X_train_aug = np.load(X_train_aug_path).astype(np.float32)
X_test_std = np.load(X_test_std_path).astype(np.float32)

# =========================
# Apply per-signal normalization
# =========================
X_train_std_norm = normalize_per_signal(X_train_std)
X_train_aug_norm = normalize_per_signal(X_train_aug)
X_test_std_norm = normalize_per_signal(X_test_std)


# =========================
# 2️ ONE HOT ENCODER on Y
# =========================
encoder = OneHotEncoder(sparse_output=False)

Y_train_aug = np.load(Y_train_aug_path)
Y_train_std = np.load(Y_train_std_path)
Y_test_std = np.load(Y_test_std_path)


Y_train_ohe = encoder.fit_transform(Y_train_aug.reshape(-1, 1))


Y_train_ohe_std = encoder.transform(Y_train_std.reshape(-1, 1))
Y_test_ohe = encoder.transform(Y_test_std.reshape(-1, 1))

# =========================
# Save outputs
# =========================
np.save(X_train_aug_path.replace(".npy", "_stdV1.npy"), X_train_aug_norm)
np.save(X_train_std_path.replace(".npy", "_stdV1.npy"), X_train_std_norm)
np.save(X_test_std_path.replace(".npy", "_stdV1.npy"), X_test_std_norm)

np.save(Y_train_aug_path.replace(".npy", "_ohe.npy"), Y_train_ohe)
np.save(Y_train_std_path.replace(".npy", "_ohe.npy"), Y_train_ohe_std)
np.save(Y_test_std_path.replace(".npy", "_ohe.npy"), Y_test_ohe)

print("Elaboration Complete.")
print(f"Shape Y Train OHE: {Y_train_ohe.shape}")
