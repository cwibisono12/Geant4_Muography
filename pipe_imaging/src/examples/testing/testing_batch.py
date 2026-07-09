#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import nn
    import utilities
    import numpy as np
    import nn_torch
    import torch

    '''
    Below is an example how to use module above to predict the outcome over the test data.
    '''

    directory_in = sys.argv[1] #directory input where the test data are
    file_model = sys.argv[2] #file model
    directory_out = sys.argv[3] #directory output to store the prediction result from the model
    option = int(sys.argv[4]) #data retrieval mode
    epoch = int(sys.argv[5]) #iteration number
    score_file = sys.argv[6] #file to store the scoring results from prediction and testing.
    layer_file = sys.argv[7] #file consisting of nodes for each layer.
    transf_file_features = sys.argv[8] #transformer file for features.
    transf_file_targets = sys.argv[9] #transformer file for target(s).
    if len(sys.argv) >= 11:
        training_mode = int(sys.argv[10]) #estimator used for training (1: scikit learn, 2: pytorch)
    if len(sys.argv) >= 12:
        file_model_2 = sys.argv[11] #2nd file_model.

    #Get the testing files list:
    file_list = utilities.retrieve_files_in_directory(directory_in)

    dim = len(file_list)

    #Get the layer from f_layer file:
    layer = nn.get_layer_from_file(layer_file)

    for k in range(dim):
        X_test, y_test = nn.get_features_append(file_list[k], arg_number = option)
    
        #Dimensionality Reduction:
        X_test_new = nn.select_features(X_test)

        #Transform the feature: (for mode = 5, 6, 7)
        if option == 5 or option == 6 or option == 7 or option == 8 or option == 9:
            X_test_new = nn.transform_feature_from_array(X_test_new, arg_mode = option)

        #Rescaling the features:
        X_test_new_scaled = nn.rescale_features(X_test_new, transf_file_features)
        if training_mode == 2:
            X_test_new_scaled = torch.from_numpy(X_test_new_scaled).float()

        #Rescaling the targets:
        y_test_scaled = nn.rescale_features(y_test, transf_file_targets)
        
        if option >= 1 and option < 7:
            if training_mode == 1:
                #Make prediction from the test file:
                result = nn.predict_outcome_from_file(X_test_new_scaled, file_model)

                #Get the model score over the test data:
                score = nn.get_model_score_from_file(X_test_new_scaled, y_test_scaled, file_model)
            if training_mode == 2:
                #Instantiate the model parameter(s):
                model_param = nn_torch.MLPRegressor(layer, ft_dim = len(X_test_new_scaled[0]), tg_dim = len(y_test_scaled[0]), activation_function='tanh')

                #Make prediction from the test file:
                result = nn_torch.predict_outcome_from_file(X_test_new_scaled, model_param, file_model)

                #Get the model score over the test data:
                y_test_scaled = torch.from_numpy(y_test_scaled).float()
                score = nn_torch.get_model_score_from_file(X_test_new_scaled, y_test_scaled, model_param, file_model)
            
            print("score:",score)

            nn.store_score_result(score_file, str(file_list[k]), layer, epoch, score, file_model)
        
        if option == 7 or option == 8 or option ==  9:
            #Split the scaled targets:
            y_test_scaled_1, y_test_scaled_2 = nn.split_target(y_test_scaled, arg_mode = option)
            del y_test_scaled

            if training_mode == 1:
                #Make prediction from the test_files:
                result_1 = nn.predict_outcome_from_file(X_test_new_scaled, file_model)
                result_2 = nn.predict_outcome_from_file(X_test_new_scaled, file_model_2)
            
                #Get the models score from the test data:
                score_1 = nn.get_model_score_from_file(X_test_new_scaled, y_test_scaled_1, file_model)
                score_2 = nn.get_model_score_from_file(X_test_new_scaled, y_test_scaled_2, file_model_2)

                print("score_1: ", score_1, "score_2: ", score_2)
            if training_mode == 2:
                y_test_scaled_1 = torch.from_numpy(np.array(y_test_scaled_1)).float()
                y_test_scaled_2 = torch.from_numpy(np.array(y_test_scaled_2)).float()

                #Instantiate the model parameter(s):
                model_param_1 = nn_torch.MLPRegressor(layer, ft_dim = len(X_test_new_scaled[0]), tg_dim = len(y_test_scaled_1[0]), activation_function='relu')
                model_param_2 = nn_torch.MLPRegressor(layer, ft_dim = len(X_test_new_scaled[0]), tg_dim = len(y_test_scaled_2[0]), activation_function='tanh')

                #Make prediction from the test_files:
                result_1 = nn_torch.predict_outcome_from_file(X_test_new_scaled, model_param_1, file_model)
                result_2 = nn_torch.predict_outcome_from_file(X_test_new_scaled, model_param_2, file_model_2)
               
                #Get the models score from the test data:
                score_1 = nn_torch.get_model_score_from_file(X_test_new_scaled, y_test_scaled_1, model_param_1, file_model)
                score_2 = nn_torch.get_model_score_from_file(X_test_new_scaled, y_test_scaled_2, model_param_2, file_model_2)

            nn.store_score_result_append(score_file, str(file_list[k]), layer, epoch, score_1, score_2, file_model, file_model_2)
            
            #Combine result into one target array:
            result = nn.combine_target(result_1, result_2, arg_mode = option)
        
        #Get the predicted file name:
        temp = utilities.retrieve_file_name(str(file_list[k]), 1, option, epoch)
        file_out_test = directory_out + temp
        
        #Inverse transform the target to revert to original spatial positions:
        result_inv = nn.inverse_transform(result, transf_file_targets)
        
        #Store the prediction from the re-trained model:
        if option >= 1 or option <= 7:
            nn.write_features(file_out_test, result_inv)
        if option == 8:
            nn.write_features(file_out_test, result_inv, coord_transform = 2)
            
        if option == 9:
            #Revert the inversed scaled target back to original spatial positions:
            result_inv_transf = nn.rescaled_target_transform(result_inv, arg_mode = option)
            del result_inv
            nn.write_features(file_out_test, result_inv_transf)
    
