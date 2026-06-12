#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import utilities
    import nn
    import random

    directory = sys.argv[1]
    file_model = sys.argv[2]
    iter_num = int(sys.argv[3]) #number of iteration (epoch)

    file_list = utilities.retrieve_files_in_directory(directory)

    dim = len(file_list)

    #Initialize the model:
    layer = (500,500,500)
    nn.model_initialize(layer, file_model)
    
    for i in range(iter_num):
        
        random.shuffle(file_list)

        for j in range(dim):
            X_train, y_train = nn.get_features_append(file_list[j], arg_number = 3)
            X_train_new = nn.select_features(X_train)
            nn.retrain_model(X_train_new, y_train, file_model)



