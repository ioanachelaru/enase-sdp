import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

NO_EPOCHS = 50
BATCH_SIZE = 64
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']
ACTIVATION = ['relu', 'sigmoid']
VERBOSE = 1


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation=ACTIVATION[0]))  # Input layer
    model.add(Dense(16, activation=ACTIVATION[0]))  # Hidden layer
    model.add(Dense(1, activation=ACTIVATION[1]))  # Output layer

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def train_model(no_clusters, target_cluster, model, x_train, y_train, x_val, y_val):
    class_weights = {0: 1, 1: 100}
    sample_weights = np.array([class_weights[label] for label in y_train.iloc[:, 0]])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), sample_weight=sample_weights, 
                        epochs=NO_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'data_to_cluster/{no_clusters}_clusters/data_cluster_{target_cluster}/model_loss.png')
    
    return model


def read_data(filepath):
    x_train = pd.read_csv(filepath + '/x_train.csv', index_col=0, header=None)
    x_val = pd.read_csv(filepath + 'x_val.csv', index_col=0, header=None)
    x_test = pd.read_csv(filepath + 'x_test.csv', index_col=0, header=None)
    y_train = pd.read_csv(filepath + 'y_train.csv', index_col=0, header=None)
    y_val = pd.read_csv(filepath + 'y_val.csv', index_col=0, header=None)
    y_test = pd.read_csv(filepath + 'y_test.csv', index_col=0, header=None)

    return x_train, x_val, x_test, y_train, y_val, y_test


def binary_classification(no_clusters, target_cluster, filename):
    x_train, x_val, x_test, y_train, y_val, y_test = read_data(filename)

    model = create_model(60)
    model = train_model(no_clusters, target_cluster, model, x_train, y_train, x_val, y_val)
    # score = model.evaluate(x_test, y_test, verbose=VERBOSE)
    # print("Test score:", score[0])
    # print("Test accuracy:", score[1])

    # return model

if __name__ == '__main__':
    binary_classification(4, 1, 'data_to_cluster/4_clusters/data_cluster_1/')
    # for i in range(4, 7):
    #     print(f'Building model for {i} clusters...')
        
    #     for ii in range(1, i+1):
    #         print(f'...building model for cluster {ii}...')
    #         binary_classification(no_clusters, target_cluster, f'data_to_cluster/{i}_clusters/data_cluster_{ii}/')