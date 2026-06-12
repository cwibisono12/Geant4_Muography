#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import nn
    import utilities

    '''
    Below is an example how to use module above to predict the outcome over the test data.
    '''

    directory_in = sys.argv[1] #directory input where the test data are
    file_model = sys.argv[2] #file model
    directory_out = sys.argv[3] #directory output to store the prediction result from the model
    option = int(sys.argv[4]) #data retrieval mode
    epoch = int(sys.argv[5]) #iteration number
    score_file = sys.argv[6] #file to store the scoring results from prediction and testing.

    #Get the testing files list:
    file_list = utilities.retrieve_files_in_directory(directory_in)

    dim = len(file_list)

    layer = (500,500,500)
    for k in range(dim):
        X_test, y_test = nn.get_features_append(file_list[k], arg_number = option)
    
        #Dimensionality Reduction:
        X_test_new = nn.select_features(X_test)

        #Make prediction from the test file
        result = nn.predict_outcome_from_file(X_test_new, file_model)

        #Get the model score over the test data
        score = nn.get_model_score_from_file(X_test_new, y_test, file_model)

        print("score:",score)

        nn.store_score_result(score_file, str(file_list[k]), layer, epoch, score, file_model)

        #Get the predicted file name:
        temp = utilities.retrieve_file_name(str(file_list[k]), 1, option, epoch)
        file_out_test = directory_out + temp

        #Store the prediction from the re-trained model:
        nn.write_features(file_out_test, result)

    

    
