import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier

NO_EPOCHS = 10
BATCH_SIZE = 32
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']
ACTIVATION = ['relu', 'sigmoid']
VERBOSE = 1


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation=ACTIVATION[0]))  # Input layer
    # model.add(Dense(16, activation=ACTIVATION[0]))  # Hidden layer
    model.add(Dense(1, activation=ACTIVATION[1]))  # Output layer

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    return model


def train_model(no_clusters, target_cluster, model, x_train, y_train, x_val, y_val):

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    # Create a dictionary to pass to the model
    class_weight = {i: w for i, w in enumerate(class_weights)}

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), class_weight=class_weight, 
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
    loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Threshold predictions to convert probabilities to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)

    # Print classification report and confusion matrix
    print("Classification Report:\n", classification_report(y_test, binary_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, binary_predictions))

    return model

if __name__ == '__main__':
    binary_classification(4, 1, 'data_to_cluster/4_clusters/data_cluster_1/')
    # for i in range(4, 7):
    #     print(f'Building model for {i} clusters...')
        
    #     for ii in range(0, i):
    #         print(f'...building model for cluster {ii}...')
    #         binary_classification(no_clusters, target_cluster, f'data_to_cluster/{i}_clusters/data_cluster_{ii}/')