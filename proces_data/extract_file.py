import os
import numpy as np
import scipy.io
import pywt
from scipy.signal import welch


# ======================================================
# MAPPING labels (Paderborn)
# ======================================================
def folder_to_label(folder: str):
    """
    return the label for the file in the folder controlling how is the synthase the name of the folder is.

    :param folder: string of the base name of the folder
    :return: a number from 0 to 3 or rise a ValueError
    """
    if folder.startswith("K") and not folder.startswith(("KI", "KA", "KB")):
        return 0  # sano
    elif folder.startswith("KI"):
        return 1  # danno interno
    elif folder.startswith("KA"):
        return 2  # danno esterno
    elif folder.startswith("KB"):
        return 3  # danno combinato
    else:
        raise ValueError(f"Cartella non riconosciuta: {folder}")

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(signal, wavelet="db4", level=3):
    features = []

    # --- Wavelet Packet Decomposition ---
    wp = pywt.WaveletPacket(
        data=signal,
        wavelet=wavelet,
        mode="symmetric",
        maxlevel=level
    )
    nodes = wp.get_level(level, order="freq")
    energies = [np.sum(n.data ** 2) for n in nodes]
    features.extend(energies)

    # --- FFT ---
    fft_power = np.abs(np.fft.fft(signal)) ** 2
    features.append(np.mean(fft_power))
    features.append(np.std(fft_power))

    # --- PSD ---
    _, Pxx = welch(signal)
    features.append(np.mean(Pxx))
    features.append(np.std(Pxx))

    return features

# ======================================================
# CONFIGURAZIONE
# ======================================================
DATA_DIR = "...../paderborn_university/dataset"

# ======================================================
# DATASET CONTAINER
# ======================================================
X_vib_raw, X_vib_wpd, y_vib = [], [], []

X_cur1_raw, X_cur1_wpd, y_cur1 = [], [], []
X_cur2_raw, X_cur2_wpd, y_cur2 = [], [], []

# ======================================================
# MAIN LOOP
# ======================================================
folders = [
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f)) and f.startswith(("K", "KI", "KA", "KB"))
]

for folder in folders:
    folder_path = os.path.join(DATA_DIR, folder)
    label = folder_to_label(folder)

    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            mat = scipy.io.loadmat(os.path.join(folder_path, file))
            main_key = [k for k in mat if not k.startswith("__")][0]
            root = mat[main_key]

            Y = root["Y"][0, 0]
            names = Y["Name"][0]
            data = Y["Data"][0]

            for i, name in enumerate(names):
                signal = data[i].squeeze()
                name = name[0]

                # ---------------- VIBRATION ----------------
                if name.startswith("vibration"):
                    X_vib_raw.append(signal)
                    X_vib_wpd.append(extract_features(signal))
                    y_vib.append(label)

                # ---------------- CURRENT PHASE 1 ----------------
                elif name == "CURRENT_PHASE_1":
                    X_cur1_raw.append(signal)
                    X_cur1_wpd.append(extract_features(signal))
                    y_cur1.append(label)

                # ---------------- CURRENT PHASE 2 ----------------
                elif name == "CURRENT_PHASE_2":
                    X_cur2_raw.append(signal)
                    X_cur2_wpd.append(extract_features(signal))
                    y_cur2.append(label)

# ======================================================
# CONVERSION ARRAY + SALVATAGGIO
# ======================================================
def save_dataset(filename, X_raw, X_wpd, y):
    np.savez_compressed(
        filename,
        X_raw=np.array(X_raw, dtype=object),
        X_wpd=np.asarray(X_wpd, dtype=np.float32),
        y=np.asarray(y, dtype=np.uint8)
    )


save_dataset("test_dataset_vibration.npz", X_vib_raw, X_vib_wpd, y_vib)
save_dataset("test_dataset_current_phase1.npz", X_cur1_raw, X_cur1_wpd, y_cur1)
save_dataset("test_dataset_current_phase2.npz", X_cur2_raw, X_cur2_wpd, y_cur2)

# ======================================================
# REPORT
# ======================================================
print("Extraction complete\n")

print(f"VIBRATION → raw: {len(X_vib_raw)}, feat: {len(X_vib_wpd)}")
print(f"CURRENT PHASE 1 → raw: {len(X_cur1_raw)}, feat: {len(X_cur1_wpd)}")
print(f"CURRENT PHASE 2 → raw: {len(X_cur2_raw)}, feat: {len(X_cur2_wpd)}")
