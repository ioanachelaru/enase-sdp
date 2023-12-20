import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from binary_classification import generate_confusion_matrix
from prepare_data_for_classification import split_data
from utilitaries import read_csv_file, write_dict_to_file

NO_EPOCHS = 40
BATCH_SIZE = 32
LOSS = 'categorical_crossentropy'  # Change the loss function for multi-class classification
LEARNING_RATE = 0.001
METRICS = ['accuracy']
ACTIVATION = 'relu'
VERBOSE = 1
DOPOUT_RATE = 0.2

def create_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation=ACTIVATION))  # Input layer
    model.add(Dense(32, activation=ACTIVATION))  # Hidden layer
    model.add(Dense(16, activation=ACTIVATION))  # Hidden layer
    model.add(Dropout(DOPOUT_RATE))  # Dropout layer
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for multi-class classification

    model.compile(loss=LOSS, optimizer=Adam(learning_rate=LEARNING_RATE), metrics=METRICS)

    return model

def train_model(no_clusters, model, x_train, y_train, x_val, y_val):
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(class_weights))

    # Convert y_train and y_val to one-hot encoding for multi-class classification
    y_train_onehot = pd.get_dummies(y_train)
    y_val_onehot = pd.get_dummies(y_val)

    history = model.fit(x_train, y_train_onehot, validation_data=(x_val, y_val_onehot), class_weight=class_weight,
                        epochs=NO_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'multiclass_classification/{no_clusters}_clusters/model_loss.png')

    return model

def label_data_for_multiclass_classification(no_clusters):
    all_data = read_csv_file('parsed_data/all_to_cluster.csv')
    clustered_bugs = read_csv_file(f'clustered_data/clustered_data_{no_clusters}_centers.csv')

    labeled_data = {key: [no_clusters] + values for key, values in all_data.items()}
    labeled_data.update(clustered_bugs)
    
    write_dict_to_file(f'multiclass_classification/{no_clusters}_clusters/x+y_{no_clusters}_clusters.csv', labeled_data)

    return labeled_data

def split_data_for_multiclass_classification(no_clusters):
    labeled_data = label_data_for_multiclass_classification(no_clusters)
    
    x = {k: v[1:] if v else None for k, v in labeled_data.items()}
    y = {k: v[0] if v else None for k, v in labeled_data.items()}

    x_df = pd.DataFrame(data=([k] + v for k, v in x.items()))
    y_df = pd.DataFrame(data=([k] + [v] for k, v in y.items()))

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_df, y_df)

    destination_dir = f'multiclass_classification/{no_clusters}_clusters/'

    x_train.to_csv(destination_dir + '/x_train.csv', index=False, header=False)
    x_val.to_csv(destination_dir + '/x_val.csv', index=False, header=False)
    x_test.to_csv(destination_dir + '/x_test.csv', index=False, header=False)
    
    y_train.to_csv(destination_dir + '/y_train.csv', index=False, header=False)
    y_val.to_csv(destination_dir + '/y_val.csv', index=False, header=False)
    y_test.to_csv(destination_dir + '/y_test.csv', index=False, header=False)

    return x_train, x_val, x_test, y_train, y_val, y_test

def read_data(filepath):
    x_train = pd.read_csv(filepath + '/x_train.csv', index_col=0, header=None)
    x_val = pd.read_csv(filepath + '/x_val.csv', index_col=0, header=None)
    x_test = pd.read_csv(filepath + '/x_test.csv', index_col=0, header=None)
    y_train = pd.read_csv(filepath + '/y_train.csv', index_col=0, header=None).squeeze()
    y_val = pd.read_csv(filepath + '/y_val.csv', index_col=0, header=None).squeeze()
    y_test = pd.read_csv(filepath + '/y_test.csv', index_col=0, header=None).squeeze()

    return x_train, x_val, x_test, y_train, y_val, y_test

def multi_class_classification(no_clusters):
    filepath = f'multiclass_classification/{no_clusters}_clusters/'
    x_train, x_val, x_test, y_train, y_val, y_test = read_data(filepath)

    num_classes = no_clusters + 1  # Add 1 for the unlabeled class

    model = create_model(60, num_classes)
    model = train_model(no_clusters, model, x_train, y_train, x_val, y_val)

    model.save(filepath + f'{no_clusters}_centers_multi_class_model.keras')

    loss, accuracy = model.evaluate(x_test, pd.get_dummies(y_test), verbose=VERBOSE)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Save classification report to a file
    report = classification_report(y_test, predicted_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filepath + "classification_report.csv")

    generate_confusion_matrix(y_test, predicted_labels, filepath + f'{no_clusters}_centers_confusion_matrix.png')

if __name__ == '__main__':
    # for i in range(4, 7):
    #     print(f'Split and write data for {i} clusters...')
    #     split_data_for_multiclass_classification(i)

    multi_class_classification(4)
    # for i in range(4, 7):
    #     print(f'Building model for {i} clusters...')
    #     multi_class_classification(i)
