import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from custom_logistic_regression import CustomLogisticRegression
from ada_boost import AdaBoost
from sklearn.metrics import confusion_matrix, classification_report



# function to load data
def load_data(file_name):
    # load dataset
    dataset = pd.read_csv(file_name)
    
    # drop the customerID column
    dataset.drop('customerID', axis = 1, inplace = True)
    
    # replace the cells with single and extra spaces with a Nan value
    dataset.replace(r'^\s*$', np.nan, regex=True, inplace = True)
    
    # convert the TotalCharges column to numeric
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])
    
    # convert the dataset to dataframe
    dataset = pd.DataFrame(dataset)
    
    return dataset


# function to find the missing columns
def find_missing_columns(dataset):
    dataset_columns_length = len(dataset.columns)
    # Store the column number of the columns with missing values in a list called missing_cols
    missing_cols = [i for i in range(dataset_columns_length) if dataset.iloc[:, i].isnull().any()]
    
    # Print columns index and names with missing values and the number of missing values
    for i in missing_cols:
        print(i, dataset.columns[i], dataset.iloc[:, i].isnull().sum())
            
    return missing_cols


def impute_missing_values(dataset, missing_columns):
    for column in missing_columns:
        column_name = dataset.columns[column]
        if dataset[column_name].dtype == 'object' or dataset[column_name].dtype == 'int64':
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            dataset[column_name] = imputer.fit_transform(dataset[column_name].values.reshape(-1, 1)).ravel()
        elif dataset[column_name].dtype == 'float64':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            dataset[column_name] = imputer.fit_transform(dataset[column_name].values.reshape(-1, 1)).ravel()
    return dataset



# function to convert the yes and no values to 1 and 0
def convert_yes_no(dataset, column_name_to_convert):
    # create a label encoder object
    label_encoder = LabelEncoder()
    
    # Apply label encoder on the column and replace the column with the encoded values
    dataset[column_name_to_convert] = dataset[column_name_to_convert].apply(label_encoder.fit_transform)
    
    # convert the dataset to a dataframe
    dataset = pd.DataFrame(dataset)
    
    return dataset

# function to convert the categorical values to numerical values using one hot encoding
def convert_categorical(dataset, column_names_to_convert):
    column_index = []
    
    # find the index of the categorical columns
    for column_name in column_names_to_convert:
        column_index.append(dataset.columns.get_loc(column_name))
        
    # convert the dataset to a numpy array
    dataset_array = dataset.values
    
    # one hot encoder object 
    one_hot_encoder = OneHotEncoder(dtype=np.int64, handle_unknown='ignore')

    # apply the one hot encoder object on the independent variable dataset
    encoded_x = one_hot_encoder.fit_transform(dataset_array[:, column_index]).toarray()
    
    # drop the original column from the dataset
    dataset_array = np.delete(dataset_array, column_index, axis = 1)
    
    # add the new columns to the dataset
    dataset_array = np.concatenate((dataset_array, encoded_x), axis = 1)
    
    
    # get the column names of the new columns
    encoded_x_column_names = one_hot_encoder.get_feature_names_out(input_features=column_names_to_convert)
    
    # drop the old column from the dataset
    dataset = dataset.drop(column_names_to_convert, axis = 1)
    
    # record the data types of each column
    original_data_types = dataset.dtypes.to_dict()
    # all the data types are int64 for encoded columns
    
    
    # record the last column number of the dataset
    last_column_number = len(dataset.columns)
    
    # reconstruct the new dataset column names
    new_column_names = list(dataset.columns[0:last_column_number-1]) + list(encoded_x_column_names)
    new_column_names.append(dataset.columns[last_column_number-1])
    
    # rearrange the columns of the dataset_array 
    # i.e. bring the dataset_array column with the last column number to the last column number of the new dataset_array
    dataset_array = np.concatenate((dataset_array[:, 0:last_column_number-1], dataset_array[:, last_column_number:], dataset_array[:, last_column_number-1:last_column_number]), axis = 1)
    
    # convert the dataset to a dataframe
    # Here the column names are the original column names and the one hot encoded column names
    # and the values are the values of the dataset array
    dataset = pd.DataFrame(data=dataset_array, columns = new_column_names)
    
    # restore the original data types of the columns
    for column_name in dataset.columns:
        if column_name in original_data_types:
            dataset[column_name] = dataset[column_name].astype(original_data_types[column_name])
        else:
            dataset[column_name] = dataset[column_name].astype('int64')
    
    return dataset


# function to divide the dataset into x and y
def divide_dataset(dataset):
    # divide the dataset into x and y
    dataset_columns_length = len(dataset.columns)
    print(dataset_columns_length)

    x = dataset.iloc[:, 0:(dataset_columns_length-1)].values
    y = dataset.iloc[:, (dataset_columns_length-1)].values
    
    return x, y

# function to split the dataset into training and testing set
def split_dataset(x, y):
    # split the dataset into training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    
    return x_train, x_test, y_train, y_test

# function to scale the numerical values
def scale_numerical_values(dataset, column_names_to_scale):
    # Scale the numerical values
    for column in column_names_to_scale:
        dataset[column] = (dataset[column] - dataset[column].min())/(dataset[column].max() - dataset[column].min())
    return dataset


# function to standardize the dataset
def pre_process(dataset):

    # Store the column number of the columns with missing values in a list called missing_cols
    missing_cols = find_missing_columns(dataset)

    dataset = impute_missing_values(dataset, missing_cols)

    # Scale the numerical values
    column_names_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    dataset = scale_numerical_values(dataset, column_names_to_scale)
    
    # Name of the column that needs to be encoded in 0s and 1s
    column_name = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

    dataset = convert_yes_no(dataset, column_name)


    # Name of the column that is a categorical variable
    categorical_col_names = ['Gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    dataset = convert_categorical(dataset, categorical_col_names)
    
    
    return dataset
    


dataset = load_data('Telco-Customer-Churn.csv')

dataset = pre_process(dataset)

x, y = divide_dataset(dataset)

x_train, x_test, y_train, y_test = split_dataset(x, y)

# model = CustomLogisticRegression(learning_rate=0.1, num_iterations=100000, early_stopping_threshold=0.3, num_features=30, verbose=True)

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)


# print()
# print('Confusion Matrix and Classification Report')
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))



model = AdaBoost(num_hypothesis=10, verbose=True, early_stopping_threshold=0.45)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print()
print('Confusion Matrix and Classification Report')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
