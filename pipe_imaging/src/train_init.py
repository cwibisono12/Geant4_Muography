#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    '''
    Below is an example how to use module to initialize the model
    '''

    file_train = sys.argv[1]
    file_model = sys.argv[2]
    option = int(sys.argv[3])

    X_train, y_train = nn.get_features_append(file_train, arg_number = option)
    #nn.write_features(file_out_train, y_train)

    dim = len(X_train)
    for i in range(dim):
        print(i, X_train[i], y_train[i])
    
    layer = (500,100,100)
    nn.initialize_model(X_train, y_train, layer, file_model, num_iter = 5000)


    

    
