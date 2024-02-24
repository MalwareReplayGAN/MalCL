import numpy as np 
from data_dir import get_ember_data


data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train, X_test, Y_test = get_ember_data(data_dir)

def data_1d(data):
    num_features = data.shape[1]
    num_rows = len(data)
    print('The number of the features in the EMBER dataset is', num_features) # 2381
    print('The number of the rows in the X_train set is', len(data)) # (X_train, 303331) (X_test, 33704)
    print('The number of the rows in the X_test set is', len(X_test))

    return data_add

X_train_conv2 = data_1d(X_train)
X_test_conv2 = data_1d(X_test)


    


