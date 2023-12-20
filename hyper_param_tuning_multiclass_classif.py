from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binary_classification import generate_confusion_matrix
from multiclass_classification import read_data


def create_model(input_dim, num_classes, learning_rate=0.001, dropout_rate=0.2, activation='relu', hidden_layers=2, neurons_per_layer=32):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation=activation))  # Input layer

    for _ in range(hidden_layers):  # Add hidden layers
        model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for multi-class classification
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def train_model(x_train, y_train, x_val, y_val, epochs=40, batch_size=32, verbose=1):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(class_weights))

    y_train_onehot = pd.get_dummies(y_train)
    y_val_onehot = pd.get_dummies(y_val)

    model = KerasClassifier(build_fn=create_model, input_dim=x_train.shape[1], num_classes=y_train_onehot.shape[1],
                            verbose=verbose, dropout_rate=0.2, hidden_layers=2, neurons_per_layer=32, learning_rate=0.001)

    param_grid = {
        'epochs': [20, 30, 40],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.1],
        'dropout_rate': [0.2, 0.4, 0.6],
        'hidden_layers': [1, 2, 3],
        'neurons_per_layer': [16, 32, 64],
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(x_train, y_train_onehot, validation_data=(x_val, y_val_onehot), class_weight=class_weight)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Print best parameters to a file
    with open('best_parameters.txt', 'w') as f:
        f.write(f"Best parameters: {grid_result.best_params_}\n")

    return grid_result.best_params_

def multi_class_classification(no_clusters):
    filepath = f'multiclass_classification/{no_clusters}_clusters/'
    x_train, x_val, x_test, y_train, y_val, y_test = read_data(filepath)

    num_classes = no_clusters + 1  # Add 1 for the unlabeled class

    best_params = train_model(x_train, y_train, x_val, y_val)

    model = create_model(60, num_classes, **best_params)
    history = model.fit(x_train, pd.get_dummies(y_train), epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'multiclass_classification/{no_clusters}_clusters/model_loss.png')


    model.save(filepath + f'{no_clusters}_centers_multi_class_model_hyperparam_tuned.keras')

    loss, accuracy = model.evaluate(x_test, pd.get_dummies(y_test), verbose=1)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(y_test, predicted_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(filepath + "classification_report_hyperparam_tuned.csv")

    generate_confusion_matrix(y_test, predicted_labels, filepath + f'{no_clusters}_centers_confusion_matrix_hyperparam_tuned.png')


if __name__ == '__main__':
    multi_class_classification(4)

