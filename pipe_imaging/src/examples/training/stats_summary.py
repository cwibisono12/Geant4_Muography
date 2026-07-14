#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import nn
    import utilities

    directory_training = sys.argv[1] #directory input where the training data are
    file_stats = sys.argv[2] #file to store summary statistics of the features
    option = int(sys.argv[3]) #data retrieval mode

    #Get the training data:
    file_list = utilities.retrieve_files_in_directory(directory_training)

    dim = len(file_list)

    #Iterate over the trainig data to obtain summary statistics for each file:
    for i in range(dim):
        X_train, y_train = nn.get_features_append(file_list[i], arg_number = option, theta_scatt = 1e-15)
        X_train_new = nn.select_features(X_train)
        
        #Store summary statistics to file_stats:
        nn.summary_statistics(X_train_new, str(file_list[i]), file_stats) 
