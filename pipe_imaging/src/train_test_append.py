#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    '''
    Below is an example how to use module for predicting first and last hits of the objects
    '''

    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_out_train = sys.argv[3]
    file_out_test = sys.argv[4]
    option = int(sys.argv[5])

    X_train, y_train = nn.get_features_append(file_train, arg_number = option)
    nn.write_features(file_out_train, y_train)

    dim = len(X_train)
    for i in range(dim):
        print(i, X_train[i], y_train[i])
    
    model_param = nn.get_parameters(X_train, y_train, layer = (100,10), num_iter = 2000)

    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    result = nn.predict_outcome(X_test, model_param)

    nn.write_features(file_out_test, result)
    nn.plot_results(result)
    
