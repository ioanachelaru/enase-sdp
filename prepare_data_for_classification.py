from sklearn.model_selection import train_test_split
import pandas as pd
from utilitaries import read_csv_file, write_dict_to_file


'''Split data (1.0.0-1.14.0) into train (80%), validation (20%) and test sets (1.15.0)'''
def split_data(x, y):
    # x_temp, x_test, y_temp, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    # x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

    # extract test set
    x_test = x[x[x.columns[0]].str.contains("1.15.0|1.14.0|1.13.0|1.12.0|1.11.0|1.10.0")]
    y_test = y[y[y.columns[0]].str.contains("1.15.0|1.14.0|1.13.0|1.12.0|1.11.0|1.10.0")]

    # remove test set from data set
    x = x[~x[x.columns[0]].str.contains("1.15.0|1.14.0|1.13.0|1.12.0|1.11.0|1.10.0")]
    y = y[~y[y.columns[0]].str.contains("1.15.0|1.14.0|1.13.0|1.12.0|1.11.0|1.10.0")]

    # split data set into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test


def write_split_data(no_clusters, target_cluster, x, y, x_train, x_val, x_test, y_train, y_val, y_test):
    destination_dir = f'data_to_cluster/{no_clusters}_clusters/data_cluster_{target_cluster}/'
    
    write_dict_to_file(destination_dir + '/x.csv', x)
    write_dict_to_file(destination_dir + '/y.csv', y)

    x_train.to_csv(destination_dir + '/x_train.csv', index=False, header=False)
    x_val.to_csv(destination_dir + '/x_val.csv', index=False, header=False)
    x_test.to_csv(destination_dir + '/x_test.csv', index=False, header=False)
    
    y_train.to_csv(destination_dir + '/y_train.csv', index=False, header=False)
    y_val.to_csv(destination_dir + '/y_val.csv', index=False, header=False)
    y_test.to_csv(destination_dir + '/y_test.csv', index=False, header=False)


'''Labels the data for binary classification cluster-of-bugs vs non-bugs'''
def label_data_for_binary_classification(no_clusters, target_cluster, target_file):
    all_data = read_csv_file('parsed_data/all_to_cluster.csv')
    clustered_bugs = read_csv_file(target_file)

    # Remove bugs from non_bugs
    for key in clustered_bugs.keys():
        all_data.pop(key, None)

    target_bugs = {k: [1] + v[1:] for k, v in clustered_bugs.items() if int(v[0]) == target_cluster}
    all_data = {key: [0] + values for key, values in all_data.items()}

    all_data.update(target_bugs)
    write_dict_to_file(f'data_to_cluster/{no_clusters}_clusters/labeled_data_for_cluster_{target_cluster}.csv', all_data)
    

def prepare_data_for_classif(no_clusters, target_cluster, file_name):
    data = read_csv_file(file_name)

    y = {k: v[0] if v else None for k, v in data.items()}
    x = {k: v[1:] if v else None for k, v in data.items()}

    x_df = pd.DataFrame(data=([k] + v for k, v in x.items()))
    y_df = pd.DataFrame(data=([k] + [v] for k, v in y.items()))

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_df, y_df)

    write_split_data(no_clusters, target_cluster, x, y, x_train, x_val, x_test, y_train, y_val, y_test)

if __name__ == '__main__':
    for i in range(4, 7):
        print(f'Preparing data for {i} clusters...')
        
        for ii in range(0, i):
            print(f'...Preparing data for cluster {ii}...')
            label_data_for_binary_classification(i, ii, f'clustered_data/clustered_data_{i}_centers.csv')
            prepare_data_for_classif(i, ii, f'data_to_cluster/{i}_clusters/labeled_data_for_cluster_{ii}.csv')