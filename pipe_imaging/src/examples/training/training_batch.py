#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities
    import nn
    import random
    import numpy as np

    directory = sys.argv[1] #directory input where the training data are
    file_model_1 = sys.argv[2] #file to store the first model or only a model
    option = int(sys.argv[3]) #data retrieval mode
    iter_num = int(sys.argv[4]) #number of maximum iteration (epoch)
    transf_file_features = sys.argv[5] #file to store the transformer to rescale the features
    transf_file_targets = sys.argv[6] #file to store the transformer to rescale the targets
    layer_file = sys.argv[7] #file to specify the nn layers
    validation_file = sys.argv[8] #validation file to verify the model performance over validation data 
    if len(sys.argv) >= 10:
        file_model_2 = sys.argv[9] #file to store the second model (if any)

    #Get the training data:
    file_list = utilities.retrieve_files_in_directory(directory)

    dim = len(file_list)

    #Initialize the transformer for the features:
    nn.scaler_initialize(transf_file_features)

    #Initialize the transformer for the targets:
    nn.scaler_initialize(transf_file_targets)
    
    #Iterate over the training data to obtain the global mean and variances for each features and target(s):
    for i in range(dim):
        X_train, y_train = nn.get_features_append(file_list[i], arg_number = option)
        X_train_new = nn.select_features(X_train)
        
        #Overwrite the last feature (for run_mode listed below):
        if option == 5 or option == 6 or option == 7:
            X_train_new = nn.transform_feature_from_array(X_train_new, arg_mode = option)

        nn.scaler_update(X_train_new, transf_file_features)
        nn.scaler_update(y_train, transf_file_targets)

    #Get the layer from layer file:
    layer = nn.get_layer_from_file(layer_file)

    #Initialize the model:
    if option == 7:
        nn.model_initialize(layer, file_model_1, activation_function = 'relu') #initialize 1st model
        nn.model_initialize(layer, file_model_2, activation_function = 'tanh') #initialize 2nd model
    if option >= 1 and option < 7:
        nn.model_initialize(layer, file_model_1, activation_function = 'tanh')

    #Extract the validation data:
    X_valid, y_valid = nn.get_features_append(validation_file, arg_number = option)
    X_valid_new = nn.select_features(X_valid)
    if option == 5 or option == 6 or option == 7:
        X_valid_new = nn.transform_feature_from_array(X_valid_new, arg_mode = option)
    X_valid_new_scaled = nn.rescale_features(X_valid_new, transf_file_features)
    y_valid_scaled = nn.rescale_features(y_valid, transf_file_targets)
    if option == 7:
        y_valid_scaled_1, y_valid_scaled_2 = nn.split_target(y_valid_scaled, arg_mode = option)

    #Initialize convergence variables:
    best_val_loss = np.inf
    flag_counter = 0
    if option == 7:
        best_val_loss_1 = np.inf
        flag_counter_1 = 0
        best_val_loss_2 = np.inf
        flag_counter_2 = 0
        converged_1 = 0
        converged_2 = 0

    global_seed = 12 #this is to match the random state listed by the estimator.

    #Iterate the model:
    for i in range(iter_num):
        
        epoch_files = file_list.copy()

        epoch_seed = global_seed + i
        random.seed(epoch_seed)

        random.shuffle(epoch_files)

        X_temp = []
        y_temp = []

        for j in range(dim):
            X_train, y_train = nn.get_features_append(epoch_files[j], arg_number = option)
            X_train_new = nn.select_features(X_train)
            if option == 5 or option == 6 or option == 7:
                X_train_new = nn.transform_feature_from_array(X_train_new, arg_mode = option)
            X_train_new_scaled = nn.rescale_features(X_train_new, transf_file_features)
            y_train_scaled = nn.rescale_features(y_train, transf_file_targets)
            
            X_temp.append(X_train_new_scaled)
            y_temp.append(y_train_scaled)

        X_temp = np.vstack(X_temp)
        y_temp = np.vstack(y_temp)
       
        if option == 7:
            if converged_1 == 1 and converged_2 == 1:
                print("Both models converged at iteration: ", i)
                break

            #Split target variables and train with two distinct models
            y_temp_1, y_temp_2  = nn.split_target(y_temp, arg_mode = option)
        
            if converged_1 == 0:
                nn.retrain_model(X_temp, y_temp_1, file_model_1)
                current_val_loss_1 = nn.calculate_mse_loss(X_valid_new_scaled, y_valid_scaled_1, file_model_1)
                converged_1, best_val_loss_1, flag_counter_1 = nn.convergence_check(
                        current_loss = current_val_loss_1,
                        best_loss = best_val_loss_1,
                        flag_counter = flag_counter_1
                        )
            
            if converged_2 == 0:
                nn.retrain_model(X_temp, y_temp_2, file_model_2)
                current_val_loss_2 = nn.calculate_mse_loss(X_valid_new_scaled, y_valid_scaled_2, file_model_2)
                converged_2, best_val_loss_2, flag_counter_2 = nn.convergence_check(
                        current_loss = current_val_loss_2,
                        best_loss = best_val_loss_2,
                        flag_counter = flag_counter_2
                        )

       
        if option >= 1 and option < 7:
            nn. retrain_model(X_temp, y_temp, file_model)
            #Assesing the current performance:
            current_val_loss = nn.calculate_mse_loss(X_valid_new_scaled, y_valid_scaled, file_model_1)

            #Convergence Checking:
            converged, best_val_loss, flag_counter = nn.convergence_check(
                    current_loss = current_val_loss,
                    best_loss = best_val_loss,
                    flag_counter = flag_counter
                    )

            if converged == 1:
                print("Model converged at iteration: ", i)
                break
