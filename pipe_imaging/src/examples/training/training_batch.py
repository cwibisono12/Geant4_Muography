#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities
    import nn
    import random
    import numpy as np

    directory = sys.argv[1] #directory input where the training data are
    file_model = sys.argv[2] #file to store the model
    option = int(sys.argv[3]) #data retrieval mode
    iter_num = int(sys.argv[4]) #number of iteration (epoch)
    transf_file = sys.argv[5] #file to store the transformer to rescale the features
    layer_file = sys.argv[6] #file to specify the nn layers
    validation_file = sys.argv[7] #validation file to verify the model performance over validation data 

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
    
    #Extract the validation data:
    X_valid, y_valid = nn.get_features_append(validation_file, arg_number = option)
    X_valid_new = nn.select_features(X_valid)
    X_valid_new_scaled = nn.rescale_features(X_valid_new, transf_file)

    #Initialize convergence variables:
    best_val_loss = np.inf
    flag_counter = 0
    random.seed(12) #this is to match the random state listed by the estimator.

    #Iterate the model:
    for i in range(iter_num):
        
        random.shuffle(file_list)

        for j in range(dim):
            X_train, y_train = nn.get_features_append(file_list[j], arg_number = option)
            X_train_new = nn.select_features(X_train)
            X_train_new_scaled = nn.rescale_features(X_train_new, transf_file)
            nn.retrain_model(X_train_new_scaled, y_train, file_model)

        
        #Assesing the current performance:
        current_val_loss = nn.calculate_mse_loss(X_valid_new_scaled, y_valid, file_model)

        #Convergence Checking:
        converged, best_val_loss, flag_counter = nn.convergence_check(
                current_loss = current_val_loss,
                best_loss = best_val_loss,
                flag_counter = flag_counter
                )

        if converged == 1:
            print("Model converged at iteration: ", i)
            break
