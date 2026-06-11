#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    '''
    Below is an example how to use module above for the first/last/average hit of the objects
    '''

    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_out = sys.argv[3]
    option = int(sys.argv[4]) #argument number for selecting the target variables (1, 2, or 3)---> see nn.get_features() function

    X_train, y_train = nn.get_features(file_train, arg_number = option)
    
    model_param = nn.get_parameters(X_train, y_train, layer = (100,10), num_iter = 2000)

    X_test, y_test = nn.get_features(file_test, arg_number = option)
    
    result = nn.predict_outcome(X_test, model_param)

    nn.write_features(file_out, result)
    nn.plot_results(result)
    
