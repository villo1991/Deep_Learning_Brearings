import random
import numpy as np

def distortion_raw_vibration(array: np.ndarray, p: float = 0.5):
    """
    Generate the distorsion to augment the data
    :param array: data to be distorted
    :param p: probability of distortion
    """
    d = array.astype(np.float32)

    if np.random.rand() < p:
        d = np.roll(d, np.random.randint(0, len(d)))
    # gaussian noise
    if np.random.rand() < p:
        noise = np.random.normal(0.0, 0.01 * np.std(d), d.shape)
        d += noise
    # amplitude scaling
    if np.random.rand() < p:
        d *= np.random.uniform(0.9, 1.1)
    return d


np.random.seed(42)
dataset = np.load("/...../paderborn_university/dataset/train_dataset_vibration.npz", allow_pickle=True)

X_v_raw = dataset["X_raw"]
Y_v_raw = dataset["y"]

idx = 0
slice_dimension = 31325
number_sample = Y_v_raw.shape[0]
new_X_dataset = np.zeros((number_sample*12,slice_dimension), dtype=np.float32)
new_Y_dataset = np.zeros((number_sample*12,), dtype=np.uint8)

for x,y in enumerate(X_v_raw):
    label = Y_v_raw[x]
    if label==1 or label==2 :
        new_X_dataset[idx, :] = y[0:slice_dimension]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension:slice_dimension * 2]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 2:slice_dimension * 3]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 3:slice_dimension * 4]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 4:slice_dimension * 5]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 5:slice_dimension * 6]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 6:slice_dimension * 7]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 7:slice_dimension * 8]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
    elif label==0:
        new_X_dataset[idx, :] = y[0:slice_dimension]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension:slice_dimension * 2]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 2:slice_dimension * 3]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 3:slice_dimension * 4]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 4:slice_dimension * 5]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 5:slice_dimension * 6]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 6:slice_dimension * 7]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 7:slice_dimension * 8]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        for _ in range(8):
            n_rand = random.randint(0,215604)
            chunk = y[n_rand:n_rand+slice_dimension]
            dist_chunk = distortion_raw_vibration(chunk, p=0.55)
            new_X_dataset[idx,:] = dist_chunk
            new_Y_dataset[idx] = label
            idx+=1
    elif label==3:
        new_X_dataset[idx, :] = y[0:slice_dimension]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension:slice_dimension * 2]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 2:slice_dimension * 3]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 3:slice_dimension * 4]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 4:slice_dimension * 5]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 5:slice_dimension * 6]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 6:slice_dimension * 7]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        new_X_dataset[idx, :] = y[slice_dimension * 7:slice_dimension * 8]
        new_Y_dataset[idx] = Y_v_raw[x]
        idx += 1
        for _ in range(13):
            n_rand = random.randint(0,215604)
            chunk = y[n_rand:n_rand+slice_dimension]
            dist_chunk = distortion_raw_vibration(chunk, p=0.5)
            new_X_dataset[idx,:] = dist_chunk
            new_Y_dataset[idx] = label
            idx+=1



new_X_dataset = new_X_dataset[:idx]
new_Y_dataset = new_Y_dataset[:idx]

control_class= np.zeros(4, dtype=np.uint16)
for x in new_Y_dataset:
    control_class[x] += 1

print(control_class)


np.save("...../paderborn_university/dataset/X_train_dataset_vibration_raw_aug_0.5s.npy",new_X_dataset)
np.save("...../paderborn_university/dataset/Y_train_raw_aug_0.5.npy",new_Y_dataset)

