import numpy as np
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import os
import pickle
import sys
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, LeakyReLU, Add,
    Input, Dense, GlobalMaxPooling1D,
    GlobalAveragePooling1D, Concatenate, Dropout,
    SpatialDropout1D, MaxPooling1D, ReLU,Flatten,LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras.layers import PReLU

# --- FUNZIONI DI UTILITÀ ---
def ConfusionMatrix(pred, y):
    cm = np.zeros((8, 8), dtype="uint16")
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
        dropout=0.5,
        padding= "same"
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.downsample = downsample
        self.useBN = useBN
        self.reg = l2(reg)
        self.dropout = dropout
        self.padding = padding

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
        y = ReLU()(y)
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
        y = ReLU()(y)
        return y


    # Kurtosis protetta
def safe_kurtosis(small_x):
    mean = tf.reduce_mean(small_x, axis=1, keepdims=True)
    centered = small_x - mean
    var = tf.reduce_mean(tf.square(centered), axis=1)
    fourth = tf.reduce_mean(tf.pow(centered, 4), axis=1)
    return fourth / (tf.square(var) + 1e-7) # Epsilon fondamentale



def ResNet1D(input_length, num_classes, useBN=True):
    inp = Input(shape=(input_length, 1))

    # --- STEM (molto leggero)
    x = Conv1D(8, 64, padding="same", strides=4, use_bias=False)(inp)
    if useBN:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.05)(x)

    # --- BLOCCO TEMPORALE A RF CRESCENTE
    x = ResidualBlock1D(8, 31, downsample=True)(x)  # 1
    x = ResidualBlock1D(8, 31, dilation_rate=2)(x)  # 2

    x = ResidualBlock1D(8, 31, downsample=True)(x)  # 3
    x = ResidualBlock1D(8, 31, dilation_rate=2)(x)  # 4
    #x = MaxPooling1D(pool_size=2)(x)
    x = ResidualBlock1D(8, 31, dilation_rate=6)(x) # 5
    x = ResidualBlock1D(8, 31, downsample=True)(x) # 6

    x = ResidualBlock1D(8, 31, dilation_rate=8)(x) # 7
    x = ResidualBlock1D(16, 31, downsample=True)(x) # 8
    x = ResidualBlock1D(16, 10, dilation_rate=4)(x) # 9
    x = ResidualBlock1D(16, 7, downsample=True)(x) # 10
    #oichi neuroni perchè hanno una vista incredibile

    # --- HEAD (per segnali impulsivi)
    #kurt_pool = Lambda(safe_kurtosis)(x)
    std_pool = Lambda(lambda x: tf.math.reduce_std(x, axis=1))(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = Concatenate()([max_pool, avg_pool, std_pool])

    x = Dense(32, kernel_regularizer=l2(1e-2))(x)
    x = LeakyReLU(0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

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

    base_path = "......./paderborn_university/dataset/5012/"
    X_train_path = base_path + "X_train_raw_5012_aug_stdV1.npy"
    y_train_path = base_path + "Y_train_raw_5012_aug_ohe.npy"
    X_test_path  = base_path + "X_test_raw_5012_noaug_stdV1.npy"
    y_test_path  = base_path + "Y_test_raw_5012_noaug_ohe.npy"

    temp_X = np.load(X_train_path, mmap_mode='r')
    X_train_size = temp_X.shape[0]
    input_length = temp_X.shape[1]
    X_test_size  = np.load(X_test_path, mmap_mode='r').shape[0]
    del temp_X

    num_classes = 8

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


    model = ResNet1D(input_length=input_length, num_classes=num_classes)

    # LEARNING RATE RIDOTTO: 1e-4 invece di 1e-3 per maggiore stabilità
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  metrics=['accuracy'])
    model.summary()

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Early Stopping: Se non migliora per 4 epoche, ferma tutto e tieni il modello migliore
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

    checkpoint_path = os.path.join("......./paderborn_university/results/5012", outdir, "best_model.keras")
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

    abs_path = os.path.join("......./paderborn_university/results/5012", outdir)
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
