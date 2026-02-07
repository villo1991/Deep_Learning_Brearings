print("ResNet_light_deep_V5.py")

import numpy as np
import tensorflow as tf
import os
import pickle
import sys
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, LeakyReLU, Add,
    Input, Dense, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Concatenate, Dropout,
    SpatialDropout1D, MaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras.layers import PReLU

#
"""
Test set accuracy: 0.7602, MCC: 0.7090
[[640   0   0   0]
 [ 16 577   0  47]
 [  0   1 639   0]
 [  3 406 141  90]]

"""



# --- FUNZIONI DI UTILITÀ ---
def ConfusionMatrix(pred, y):
    cm = np.zeros((4, 4), dtype="uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

# --- BLOCCO RESIDUO (CON REGOLARIZZAZIONE L2) ---
class ResidualBlock1D:
    def __init__(
        self,
        filters,
        kernel_size=31,
        dilation_rate=1,
        downsample=False,
        useBN=True,
        reg=1e-4,
        dropout=0.1
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.downsample = downsample
        self.useBN = useBN
        self.reg = l2(reg)
        self.dropout = dropout

    def __call__(self, x):
        stride = 2 if self.downsample else 1
        dilation = 1 if self.downsample else self.dilation_rate

        # Shortcut
        shortcut = x
        if x.shape[-1] != self.filters or self.downsample:
            shortcut = Conv1D(
                self.filters, 1, strides=stride,
                padding="same", use_bias=False,
                kernel_regularizer=self.reg
            )(shortcut)
            if self.useBN:
                shortcut = BatchNormalization()(shortcut)

        # Conv 1
        y = Conv1D(
            self.filters,
            self.kernel_size,
            strides=stride,
            dilation_rate=dilation,
            padding="same",
            use_bias=False,
            kernel_regularizer=self.reg
        )(x)
        if self.useBN:
            y = BatchNormalization()(y)
        y = LeakyReLU(0.1)(y)
        y = SpatialDropout1D(self.dropout)(y)

        # Conv 2
        y = Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="same",
            use_bias=False,
            kernel_regularizer=self.reg
        )(y)
        if self.useBN:
            y = BatchNormalization()(y)

        y = Add()([y, shortcut])
        y = LeakyReLU(0.1)(y)
        return y



# --- ARCHITETTURA MODELLO (PIÙ LEGGERA E ROBUSTA) ---
def ResNet1D(input_length, num_classes, useBN=True):
    inp = Input(shape=(input_length, 1))

    # --- STEM (molto leggero)
    x = Conv1D(4, 20, padding="same", use_bias=False)(inp)  # RF = 1+ 20 AUMENTATO IL PRIMO kernel_size
    if useBN:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    #x = PReLU()(x)

    # --- BLOCCO TEMPORALE A RF CRESCENTE
    x = ResidualBlock1D(4, 31, dilation_rate=1)(x)  # RF = 15 + 30 *1 = 45
    x = ResidualBlock1D(4, 31, dilation_rate=2)(x)  # RF = 45 + 30 *2 = 105

    x = ResidualBlock1D(8, 31, dilation_rate=4)(x)  # RF = 105 + 30 *4 = 225
    x = ResidualBlock1D(8, 31, dilation_rate=8)(x)  # RF = 225 + 30 *8 = 465
    #x = MaxPooling1D(pool_size=2)(x)
    x = ResidualBlock1D(8, 31, dilation_rate=8)(x) # RF = 465 + 30 *16 = 945   #1905 # 465 + 30 *8 =705
    x = ResidualBlock1D(8, 31, dilation_rate=16)(x) # RF = 945 + 30 *32 = 1905  #3825 # 705 + 30*8 = 945

    x = ResidualBlock1D(16, 31, dilation_rate=32)(x) # RF = 1905 + 30 *64 = 3825 # 7665  #945 + 30*16 = 1425
    x = ResidualBlock1D(16, 31, dilation_rate=64)(x) # RF = 3825 + 30 *128 = 7665 # 15345 #1425+30*16 = 1905
    x = ResidualBlock1D(16, 31, dilation_rate=32)(x) # RF = 7665 + 30 *256 = 15345 # 30705 # 1905+30*32 = 2865
    x = ResidualBlock1D(16, 31, dilation_rate=16)(x) # RF = 15345 + 30*256 = 23025 # 2865 +30*64
    #oichi neuroni perchè hanno una vista incredibile

    # --- HEAD (per segnali impulsivi)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate()([max_pool, avg_pool])

    x = Dense(128, kernel_regularizer=l2(1e-2))(x)
    x = LeakyReLU(0.20)(x)
    #x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inp, out)


# --- GENERATORI DATI ---
def gen_batches(X_path, y_path, batch_size):
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    n = X.shape[0]
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield X[batch_idx][:, :, np.newaxis].astype(np.float32), y[batch_idx].astype(np.float32)

def gen_batches_data(X_path, y_path, batch_size):
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    n = X.shape[0]
    indices = np.arange(n)
    while True:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield X[batch_idx][:, :, np.newaxis].astype(np.float32), y[batch_idx].astype(np.float32)

def test_batches(X_path, batch_size):
    X = np.load(X_path, mmap_mode='r')
    n = X.shape[0]
    indices = np.arange(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx][:, :, np.newaxis].astype(np.float32)

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py BATCH EPOCHS USE_BN OUTDIR INFER_BATCH")
        sys.exit(1)

    batch_dimension = int(sys.argv[1])
    epoch = int(sys.argv[2])
    useBN = bool(int(sys.argv[3]))
    outdir = sys.argv[4]
    inference_batch_dimension = int(sys.argv[5])

    base_path = "/home/villo/projects/bearings_project/paderborn_university/dataset/"
    X_train_path = base_path + "X_train_processed_aug.npy"
    y_train_path = base_path + "Y_train_raw_aug_0.5_ohe.npy"
    X_test_path  = base_path + "X_test_processed.npy"
    y_test_path  = base_path + "Y_test_raw_0.5_ohe.npy"

    temp_X = np.load(X_train_path, mmap_mode='r')
    X_train_size = temp_X.shape[0]
    input_length = temp_X.shape[1]
    X_test_size  = np.load(X_test_path, mmap_mode='r').shape[0]
    del temp_X

    num_classes = 4

    train_dataset = tf.data.Dataset.from_generator(
        lambda: gen_batches(X_train_path, y_train_path, batch_dimension),
        output_signature=(
            tf.TensorSpec(shape=(None, input_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )).repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: gen_batches_data(X_test_path, y_test_path, batch_dimension),
        output_signature=(
            tf.TensorSpec(shape=(None, input_length, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = int(np.ceil(X_train_size / batch_dimension))
    val_steps = int(np.ceil(X_test_size / batch_dimension))

    model = ResNet1D(input_length=input_length, num_classes=num_classes, useBN=useBN)

    # LEARNING RATE RIDOTTO: 1e-4 invece di 1e-3 per maggiore stabilità
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                  metrics=['accuracy'])
    model.summary()

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Early Stopping: Se non migliora per 4 epoche, ferma tutto e tieni il modello migliore
    early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)

    checkpoint_path = os.path.join("/home/villo/projects/bearings_project/paderborn_university/results", outdir, "best_model.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    history = model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              epochs=epoch,
              verbose=1,
              validation_data=val_dataset,
              validation_steps=val_steps,
              callbacks=[lr_scheduler, model_checkpoint, early_stop]
              )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_batches(X_test_path, inference_batch_dimension),
        output_signature=(
            tf.TensorSpec(shape=(None, input_length, 1), dtype=tf.float32)
        )).prefetch(tf.data.AUTOTUNE)

    # Carica il pesi migliori (restore_best_weights in EarlyStopping lo fa, ma per sicurezza usiamo il checkpoint)
    model.load_weights(checkpoint_path)

    y_test_true = np.argmax(np.load(y_test_path).astype(np.float32), axis=1)
    pred_raw = model.predict(test_dataset, verbose=1)
    pred_label = np.argmax(pred_raw, axis=1)

    abs_path = os.path.join("/home/villo/projects/bearings_project/paderborn_university/results", outdir)
    os.makedirs(abs_path, exist_ok=True)

    d = [history.history['loss'], history.history['val_loss'],
         1.0 - np.array(history.history['accuracy']),
         1.0 - np.array(history.history['val_accuracy'])]

    pickle.dump(d, open(os.path.join(abs_path, "results.pkl"), "wb"))

    cm, acc = ConfusionMatrix(pred_label, y_test_true)
    mcc = matthews_corrcoef(y_test_true, pred_label)

    s = 'Test set accuracy: %0.4f, MCC: %0.4f' % (acc, mcc)
    with open(os.path.join(abs_path, "accuracy_mcc.txt"), "w") as f:
        f.write(s + "\n")

    np.save(os.path.join(abs_path, "confusion_matrix.npy"), cm)
    np.save(os.path.join(abs_path, "predictions.npy"), pred_raw)

    print("\n" + s)
    print(cm)
