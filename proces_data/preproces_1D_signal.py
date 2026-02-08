import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def array_2D_processed(BASE_DIR: str, x: str, fs: int) -> np.ndarray:
    array_2D = np.load(os.path.join(BASE_DIR,x)).astype(np.float32)
    array_b_a = np.gradient(array_2D, 1 / fs, axis=1).astype(np.float32)
    return array_b_a

BASE_DIR = "......./bearings_project/paderborn_university/dataset"
train_raw = "X_train_dataset_vibration_raw_0.5.npy"
train_raw_aug = "X_train_dataset_vibration_raw_aug_0.5s.npy"
test_raw = "X_test_dataset_vibration_raw_0.5.npy"


scaler = StandardScaler()
train_raw_array = np.load(os.path.join(BASE_DIR,train_raw)).astype(np.float32)
scaler.fit(train_raw_array)
del train_raw_array

array_2D = array_2D_processed(BASE_DIR, train_raw, 64000)
array_2D = scaler.transform(array_2D)
np.save(os.path.join(BASE_DIR, "X_train_processed.npy"), array_2D)

array_2D = array_2D_processed(BASE_DIR, train_raw_aug, 64000)
array_2D = scaler.transform(array_2D)
np.save(os.path.join(BASE_DIR, "X_train_processed_aug.npy"), array_2D)

array_2D = array_2D_processed(BASE_DIR, test_raw, 64000)
array_2D = scaler.transform(array_2D)
np.save(os.path.join(BASE_DIR, "X_test_processed.npy"), array_2D)
