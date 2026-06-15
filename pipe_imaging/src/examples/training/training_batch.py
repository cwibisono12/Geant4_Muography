#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities
    import nn
    import random

    directory = sys.argv[1] #directory input where the training data are
    file_model = sys.argv[2] #file to store the model
    option = int(sys.argv[3]) #data retrieval mode
    iter_num = int(sys.argv[4]) #number of iteration (epoch)
    transf_file = sys.argv[5] #file to store the transformer to rescale the features
    layer_file = sys.argv[6] #file to specify the nn layers

    file_list = utilities.retrieve_files_in_directory(directory)

    dim = len(file_list)

    #Initialize the transformer:
    nn.scaler_initialize(transf_file)
    
    #Iterate over the training data to obtain the global mean and variances for each features:
    for i in range(dim):
        X_train, y_train = nn.get_features_append(file_list[i], arg_number = option)
        X_train_new = nn.select_features(X_train)
        nn.scaler_update(X_train_new, transf_file)

    #Get the layer from layer file:
    layer = nn.get_layer_from_file(layer_file)

    #Initialize the model:
    nn.model_initialize(layer, file_model)
    
    for i in range(iter_num):
        
        random.shuffle(file_list)

        for j in range(dim):
            X_train, y_train = nn.get_features_append(file_list[j], arg_number = option)
            X_train_new = nn.select_features(X_train)
            X_train_new_scaled = nn.rescale_features(X_train_new, transf_file)
            nn.retrain_model(X_train_new_scaled, y_train, file_model)



