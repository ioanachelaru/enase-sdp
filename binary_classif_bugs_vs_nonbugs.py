from sklearn.discriminant_analysis import StandardScaler
from utilitaries import read_csv_file
from binary_classification import create_model, generate_confusion_matrix, read_data, NO_EPOCHS, BATCH_SIZE, VERBOSE
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from prepare_data_for_classification import split_data

def prepare_data():
    ALL_DATA = read_csv_file('parsed_data/all_to_cluster.csv')
    BUGS = read_csv_file('parsed_data/bugs_to_cluster.csv')

    LABELED_BUGS = {key: ([1] + values) for key, values in BUGS.items()}
    LABELED_DATA = {key: [0] + values for key, values in ALL_DATA.items()}
    LABELED_DATA.update(LABELED_BUGS)

    y = {k: v[0] if v else None for k, v in LABELED_DATA.items()}
    x = {k: v[1:] if v else None for k, v in LABELED_DATA.items()}
    x_df = pd.DataFrame(data=([k] + v for k, v in x.items()))
    y_df = pd.DataFrame(data=([k] + [v] for k, v in y.items()))

    x_df.to_csv('bugs_vs_nonbugs/x.csv', index=False, header=False)
    y_df.to_csv('bugs_vs_nonbugs/y.csv', index=False, header=False)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_df, y_df)
    x_train.to_csv('bugs_vs_nonbugs/x_train.csv', index=False, header=False)
    y_train.to_csv('bugs_vs_nonbugs/y_train.csv', index=False, header=False)
    x_val.to_csv('bugs_vs_nonbugs/x_val.csv', index=False, header=False)
    y_val.to_csv('bugs_vs_nonbugs/y_val.csv', index=False, header=False)
    x_test.to_csv('bugs_vs_nonbugs/x_test.csv', index=False, header=False)
    y_test.to_csv('bugs_vs_nonbugs/y_test.csv', index=False, header=False)


def binary_classification():
    x_train, x_val, x_test, y_train, y_val, y_test = read_data('bugs_vs_nonbugs/')

    model = create_model(60)

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
    plt.savefig('bugs_vs_nonbugs/model_loss.png')

    model.save('bugs_vs_nonbugs/binary_classif.keras')

    loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    y_pred = model.predict(x_test)

    # Threshold predictions to convert probabilities to binary predictions
    binary_predictions = (y_pred > 0.5).astype(int)

    generate_confusion_matrix(y_test, binary_predictions, f'bugs_vs_nonbugs/confusion_matrix.png')

    # Compute ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred)

    # Compute specificity
    tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
    specificity = tn / (tn + fp)

    # Save classification report to a file
    report = classification_report(y_test, binary_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Add specificity and AUC to the report
    report_df['specificity'] = specificity
    report_df['ROC-AUC'] = roc_auc

    report_df.to_csv(f'bugs_vs_nonbugs/classification_report.csv', index=False)


if __name__ == '__main__':
    # prepare_data()
    binary_classification()