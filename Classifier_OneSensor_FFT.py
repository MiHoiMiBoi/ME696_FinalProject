import numpy as np
import pandas as pd
import random
import scipy
import FeatureExtract

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def get_xy(folder, label):
    data_list = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(folder + filename)
            data = df['data']
            # data_fft = np.fft(data)
            data_list.append(np.fft.fft(data).tolist())
            labels.append(label)
    return data_list, labels

def classify(method, rs=32, rs2 = 2):
    # leak_data, leak_features, leak_labels = get_xy(folder='C:/Users/Eric/Desktop/ME696_FinalProject/data/Leak/',label=0)
    # no_leak_data, no_leak_features, no_leak_labels = get_xy(folder='C:/Users/Eric/Desktop/ME696_FinalProject/data/No Leak/',label=1)
    leak_data, leak_labels = get_xy(folder='D:/Eric/Documents/ME696_FinalProject/data/Leak2/',label=1)
    no_leak_data, no_leak_labels = get_xy(folder='D:/Eric/Documents/ME696_FinalProject/data/No Leak2/',label=0)
    leak_data2, leak_labels2 = get_xy(folder='D:/Eric/Documents/ME696_FinalProject/data/Leak1_2/',label=1)
    no_leak_data2, no_leak_labels2 = get_xy(folder='D:/Eric/Documents/ME696_FinalProject/data/No Leak1_2/',label=0)

    data_list = leak_data + no_leak_data
    label_list = leak_labels + no_leak_labels
    data_list2 = leak_data2 + no_leak_data2
    label_list2 = leak_labels2 + no_leak_labels2
    cnn(data_list,label_list,data_list2,label_list2,rs,rs2)  #,rs

def my_evaluate(truth,y_pred):
    print('Accuracy:\t',round(accuracy_score(truth,y_pred)*100,2),'%')
    precis = precision_score(truth,y_pred, average=None)
    print('Precision:\tNo Leak:',round(precis[0]*100,2),'%','\tLeak:',round(precis[1]*100,2),'%')
    recall = recall_score(truth,y_pred, average=None)
    print('Recall:\t\tNo Leak:',round(recall[0]*100,2),'%','\tLeak:',round(recall[1]*100,2),'%')
    f1 = f1_score(truth,y_pred, average=None)
    print('F1:\t\t\tNo Leak:',round(f1[0]*100,2),'%','\tLeak:',round(f1[1]*100,2),'%')
    return round(accuracy_score(truth,y_pred)*100,2)

def cnn_preprocess(data, labels, padvalue = 0):
    #print([*mylist, *[padvalue]*(N-len(mylist))])
    N = max([len(datapoint) for datapoint in data])
    print('N = ',N)

    padded_data = []
    for datapoint in data:
        pad_size = N-len(datapoint)
        padded = [*datapoint, *[padvalue]*pad_size]
        padded_data.append(padded)
    print(np.array(padded_data).shape)
    # padded_data = scipy.ndimage.gaussian_filter(padded_data, (1,24))

    onehot = to_categorical(labels)
    print(np.array(onehot).shape)

    return N, padded_data, onehot

def cnn(data, labels, data2, labels2, rs, rs2, epochs = 200, batch_size = 24):    #, rs
    rand1 = random.randint(0, 400000000)
    rand2 = random.randint(0, 400000000)
    rand3 = random.randint(0, 400000000)
    print(rand1)
    print(rand2)
    print(rand3)
    np.random.seed(rand1)
    tf.random.set_seed(rand2)
    # np.random.seed(rs2)
    # tf.random.set_seed(rs2)

    N, padded_data, onehot = cnn_preprocess(data,labels)
    # N2, padded_data2, onehot2 = cnn_preprocess(data2,labels2)
    X_train, X_test, y_train, y_test = train_test_split(padded_data, onehot, test_size = 0.3, random_state = rand3)   #, random_state = rs

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # X_train = np.array(padded_data)
    # X_test = np.array(padded_data2)
    # y_train = np.array(onehot)
    # y_test = np.array(onehot2)

    print(len(X_train),' - ',len(X_test),' - ',len(X_train) + len(X_test))

    model = Sequential([
                        Conv1D(filters = 32, kernel_size = 100, activation = 'relu', input_shape = (N, 1)),
                        MaxPooling1D(pool_size = 3),
                        Conv1D(filters = 32, kernel_size = 100, activation = 'relu'),
                        MaxPooling1D(pool_size = 3),
                        # Dense(24, activation = 'relu',input_shape = (N, 1)),
                        Flatten(),
                        Dense(12, activation = 'relu'),
                        Dense(2, activation = 'softmax')])

    opt = Adam(learning_rate = 1e-3)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    model.summary()
    # data3, labels3 = get_xy(folder='D:/Eric/Documents/ME696_FinalProject/data/k/', label=0)
    # N2, padded_data2, onehot2 = cnn_preprocess(data3, labels3)
    # new_data = np.array(padded_data2)
    # new_answers = np.array(labels3)
    for i in range(epochs):
        model.fit(X_train, y_train, epochs = 1, batch_size = batch_size, verbose=1)
        print(X_test.shape)
        y_pred_prob = model.predict(X_test)
        y_pred, truth = [], []
        for prob in y_pred_prob:
            y_pred.append(np.argmax(prob))
        for prob in y_test:
            truth.append(np.argmax(prob))
        print(i)
        my_evaluate(truth, y_pred)

        # print(new_data.shape)
        # pred = model.predict(new_data)
        # y_pred, truth = [], []
        # for prob in pred:
        #     y_pred.append(np.argmax(prob))
        # for prob in new_answers:
        #     truth.append(np.argmax(prob))
        # my_evaluate(truth, y_pred)
    # return my_evaluate(truth, y_pred)

if __name__ == '__main__':
    scores = []
    for i in range(1):
        scores.append(classify('cnn'))
    # print(np.mean(np.array(scores)))