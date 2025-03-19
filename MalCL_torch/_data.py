from data_ import get_ember_train_data, get_ember_test_data



def dataset(config):

    X_train, Y_train = get_ember_train_data(config.train_data)
    X_test, Y_test = get_ember_test_data(config.test_data)

    if len(X_train[0]) != len(X_test[0]):
        print("error!")
        return
    
    config.feats_length= len(X_train[0])

    # config.num_training_samples = 303331

    return X_train, Y_train, X_test, Y_test
