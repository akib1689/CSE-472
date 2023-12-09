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
def load_telco_data(file_name):
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

# funciton to load data
def load_credit_card_data(file_name):
    # load dataset
    dataset = pd.read_csv(file_name)
    
    # replace the cells with single and extra spaces with a Nan value
    dataset.replace(r'^\s*$', np.nan, regex=True, inplace = True)
    
    # convert the dataset to dataframe
    dataset = pd.DataFrame(dataset)
    
    return dataset

# function to load data
def load_adult_data(file_name, name_of_columns):
    """
    Load data from a file into a pandas dataframe
    :param file_name: name of file to load
    :return: pandas dataframe
    """
    dataset = pd.read_csv(file_name, header=None, names=name_of_columns)
    
    
    # replace the cells with ' ?' with NaN
    dataset = dataset.replace(' ?', np.NaN)
    
    # replce the cells with single or extra spaces with NaN
    dataset.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
    
    # drop the education column as it is redundant
    dataset.drop('education', axis=1, inplace=True)
    
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
    for column_name in column_names_to_scale:
        # max and min values
        max_value = dataset[column_name].max()
        min_value = dataset[column_name].min()
        
        # scale the values
        dataset[column_name] = (dataset[column_name] - min_value)/(max_value - min_value)
    return dataset

# function to standardize the dataset
def pre_process_telco(dataset):

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

def scale_credit_card_data(dataset):
    # get the column names of each column without the last column
    columns = dataset.columns[:-1]
    # for each column, get the max and min values
    for column in columns:
        max_value = dataset[column].max()
        min_value = dataset[column].min()
        dataset[column] = (dataset[column] - min_value)/(max_value - min_value)
        
    return dataset

def convert_label_to_binary(dataset, label_column_name, options):
    dataset[label_column_name] = dataset[label_column_name].map(options)
    return dataset

def align_columns(train, test):
    # Get missing columns in the training test
    missing_cols = set(train.columns) - set(test.columns)

    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test[c] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]

    return train, test

def pre_process_adult(train_dataset, test_dataset):
    train_missing_cols = find_missing_columns(train_dataset)
    test_missing_cols = find_missing_columns(test_dataset)
    
    train_dataset = impute_missing_values(train_dataset, train_missing_cols)
    test_dataset = impute_missing_values(test_dataset, test_missing_cols)
    
    # convert the label column to 0s and 1s
    train_dataset = convert_label_to_binary(train_dataset, 'income', {' <=50K': 0, ' >50K': 1})
    test_dataset = convert_label_to_binary(test_dataset, 'income', {' <=50K.': 0, ' >50K.': 1})
    
    categorical_col_names = ['workclass', 'marital-status', 'relationship', 'occupation', 'race', 'sex', 'native-country']
    
    train_dataset = convert_categorical(train_dataset, categorical_col_names)
    test_dataset = convert_categorical(test_dataset, categorical_col_names)
    train_dataset, test_dataset = align_columns(train_dataset, test_dataset)
    
    
    
    numerical_col_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    train_dataset = scale_numerical_values(train_dataset, numerical_col_names)
    test_dataset = scale_numerical_values(test_dataset, numerical_col_names)
    
    
    return train_dataset, test_dataset
    
    
def pre_process_credit_card(dataset):
    # Adjust for the class imbalance
    df1 = dataset[dataset['Class'] == 1]
    df0 = dataset[dataset['Class'] == 0].sample(frac = 0.01)
    dataset = pd.concat([df1, df0])
    dataset = scale_credit_card_data(dataset)
    return dataset

def report(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('Recall: ', TPR)
    
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('Specificity: ', TNR)
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('Precision: ', PPV)
    
    # False discovery rate
    FDR = FP/(TP+FP)
    print('False discovery rate: ', FDR)
    
    # F1 score
    F1 = 2*(PPV*TPR)/(PPV+TPR)
    print('F1 score: ', F1)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Accuracy: ', ACC)
    


# take an input from the user the number of the dataset to be used
number_of_dataset = int(input('Enter the number of the dataset to be used: '))

# switch case for 1,2,3
if number_of_dataset == 1:
    # load the telco dataset
    dataset = load_telco_data('Telco-Customer-Churn.csv')
    dataset = pre_process_telco(dataset)
    x, y = divide_dataset(dataset)
    x_train, x_test, y_train, y_test = split_dataset(x, y)
elif number_of_dataset == 2:
    # load the adult dataset
    name_of_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    dataset = load_adult_data('adult.data', name_of_columns)

    test_dataset = load_adult_data('adult.test', name_of_columns)
    
    dataset, test_dataset = pre_process_adult(dataset, test_dataset)
    
    x_train, y_train = divide_dataset(dataset)
    x_test, y_test = divide_dataset(test_dataset)
    
elif number_of_dataset == 3:
    # load the credit card dataset
    dataset = load_credit_card_data('creditcard.csv')
    
    dataset = pre_process_credit_card(dataset)
    
    x, y = divide_dataset(dataset)
    
    x_train, x_test, y_train, y_test = split_dataset(x, y)
    
else:
    print('Invalid input')
    exit()
    
    
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)


# take an input from the user the number of the features to be used
number_of_features = int(input('Enter the number of features to be used: '))

# take an input from the user the number of the iterations to be used
number_of_iterations = int(input('Enter the number of iterations to be used: '))

# take an input from the user the learning rate to be used
learning_rate = float(input('Enter the learning rate to be used: '))
# take an input from the user the early stopping threshold to be used
early_stopping_threshold = float(input('Enter the early stopping threshold to be used: '))

classifier = CustomLogisticRegression(num_iterations=number_of_iterations, learning_rate=learning_rate, early_stopping_threshold=early_stopping_threshold, num_features=number_of_features, verbose=False)

# fit the model
classifier.fit(x_train, y_train)

# predict the values
y_pred_test = classifier.predict(x_test)
y_pred_train = classifier.predict(x_train)


print('------------------Logistic Regression------------------')
print('-------------On Test Data-------------')
# print the confusion matrix
print('\nConfusion matrix: \n')
cf_matrix = confusion_matrix(y_test, y_pred_test)
print(cf_matrix)

# print the classification report
print('\nClassification report: \n')
print(classification_report(y_test, y_pred_test, digits=6))
report(cf_matrix)
print('-------------On Train Data-------------')
# print the confusion matrix
print('\nConfusion matrix: \n')
cf_matrix = confusion_matrix(y_train, y_pred_train)
print(cf_matrix)

# print the classification report
print('\nClassification report: \n')
print(classification_report(y_train, y_pred_train, digits=6))
report(cf_matrix)


# AdaBoost

# take an input from the user the number of the estimators to be used
number_of_estimators = int(input('Enter the number of estimators to be used: '))

# re use the early stopping threshold from the previous classifier
adaboost_classifier = AdaBoost(num_hypothesis=number_of_estimators, verbose=False, early_stopping_threshold=early_stopping_threshold)

# fit the model
adaboost_classifier.fit(x_train, y_train)

# predict the values
y_pred_test = adaboost_classifier.predict(x_test)
y_pred_train = adaboost_classifier.predict(x_train)


print('------------------AdaBoost------------------')
print('-------------On Test Data-------------')
# print the confusion matrix
print('\nConfusion matrix: \n')
cf_matrix = confusion_matrix(y_test, y_pred_test)
print(cf_matrix)

# print the classification report
print('\nClassification report: \n')
print(classification_report(y_test, y_pred_test, digits=6))
report(cf_matrix)
print('-------------On Train Data-------------')
# print the confusion matrix
print('\nConfusion matrix: \n')
cf_matrix = confusion_matrix(y_train, y_pred_train)
print(cf_matrix)

# print the classification report
print('\nClassification report: \n')
print(classification_report(y_train, y_pred_train, digits=6))
report(cf_matrix)

    
    
    
    