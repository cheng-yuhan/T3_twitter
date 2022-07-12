from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import LSTM, Dense, Dropout,concatenate,Flatten
from tensorflow.keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras
from sklearn.metrics import recall_score, precision_score, f1_score
from keras import backend as K
from tensorflow.keras import optimizer

class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

def outage_model():
    input_1 = Input(shape=(10,6),name='input1')
    input_2 = Input(shape=(9),name='input2')

    x1 = LSTM(units = 8,kernel_initializer='random_normal', bias_initializer='zeros',return_sequences=True)(input_1)
    x1 = Flatten()(x1)
    #x1 = Dropout(0.3)(x1)
    x1 = Dense(units = 8, kernel_initializer='random_normal', bias_initializer='zeros')(x1)
    x1 = layers.LeakyReLU()(x1)

    x2 = Dense(units = 8, kernel_initializer='random_normal', bias_initializer='zeros')(input_2)
    x2 = layers.LeakyReLU()(x2)
    x = concatenate([x1,x2])
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs= [input_1,input_2], outputs = [output])
    model.summary()

    return model

if __name__ == "__main__":
    df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train_data.csv')
    df = df[~(np.isnan(df).any(axis=1))]
    values = df.values
    values = values.astype('float32')
    np.random.shuffle(values)
    data_1 = values[:, 1:10]
    data_2 = values[:, 10:]
    label = values[:, 0]

    print(np.any(np.isnan(values)))

    for i in range(label.shape[0]):
        if label[i] < 0.5:
            label[i] = 0
        elif label[i] >= 0.5 and label[i] < 5:
            label[i] = 1
        else:
            label[i] = 2
    encoder = LabelEncoder()
    encoder.fit(label)
    encode_y = encoder.transform(label)
    y = np_utils.to_categorical(encode_y)
    print(y.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_1 = scaler.fit_transform(data_1)
    data_2 = scaler.fit_transform(data_2)
    data_2 = np.reshape(data_2, (-1, 10, 6))
    print(data_1.shape)
    print(data_2.shape)
    test_label = y[:500, :]
    train_label = y[1500:, :]
    test_data1 = data_1[:500]
    train_data1 = data_1[1500:]
    test_data2 = data_2[:500]
    train_data2 = data_2[1500:]
    print(test_label.shape)
    print(test_data1.shape)
    print(test_data2.shape)

    model = outage_model()

    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-3), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit([train_data2, train_data1], train_label, epochs=500, batch_size=100,
                        validation_data=([test_data2, test_data1], test_label),
                        callbacks=Metrics(valid_data=([test_data2, test_data1], test_label)))
