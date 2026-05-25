#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    '''
    Below is an example how to use module for predicting first and last hits of the objects
    '''

    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_out = sys.argv[3]

    X_train, y_train = nn.get_features_append(file_train)
    
    dim = len(X_train)
    for i in range(dim):
        print(i, X_train[i], y_train[i])
    
    model_param = nn.get_parameters(X_train, y_train)

    X_test, y_test = nn.get_features_append(file_test)
    
    result = nn.predict_outcome(X_test, model_param)

    nn.write_features(file_out, result)
    nn.plot_results(result)
    
