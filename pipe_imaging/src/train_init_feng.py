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

    #Dimensionality Reduction:
    X_train_new = nn.select_features(X_train)

    #Preprocessing the features:
    X_train_scaled = nn.preprocess_selected_features(X_train_new)

    dim = len(X_train_new)
    for i in range(dim):
        print(i, X_train_scaled[i], y_train[i])
    
    layer = (500, 500, 500)
    nn.initialize_model(X_train_scaled, y_train, layer, file_model, num_iter = 3000)


    

    
