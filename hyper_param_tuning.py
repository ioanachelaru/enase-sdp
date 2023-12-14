from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from binary_classification import read_data
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def create_model(learning_rate=0.01, units=32, activation='relu'):
    model = Sequential()
    model.add(Dense(units, input_dim=60, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0, activation='relu', learning_rate=0.01, units=32)

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'units': [16, 32, 64],
        'activation': ['relu', 'sigmoid']
    }


    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {i: w for i, w in enumerate(class_weights)}

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), class_weight=class_weight)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result.best_params_

if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test = read_data('data_to_cluster/4_clusters/data_cluster_1/')

    best_params = tune_hyperparameters(x_train, y_train, x_val, y_val)
    print(best_params)