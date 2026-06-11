#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import nn

    '''
    Below is an example how to use module above to retrain new data and predict the outcome of the test data.
    '''

    file_retrain = sys.argv[1]
    file_model = sys.argv[2]
    file_test = sys.argv[3]
    file_out_test = sys.argv[4]
    option = int(sys.argv[5])

    X_train, y_train = nn.get_features_append(file_retrain, arg_number = option)
    
    #Dimensionality Reduction:
    X_train_new = nn.select_features(X_train)

    dim = len(X_train_new)
    for i in range(dim):
        print(i, X_train_new[i], y_train[i])
    
    layer = (500,500,500,100)
    nn.retrain_model(X_train_new, y_train, file_model)
    
    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    #Dimensionality Reduction:
    X_test_new = nn.select_features(X_test)

    result = nn.predict_outcome_from_file(X_test_new, file_model)

    score = nn.get_model_score_from_file(X_test_new, y_test, file_model)

    print("score:",score)

    nn.store_score_result('../../corr_files/retrain_test_score_reduce_dim.csv', file_test, layer, 2000, score)

    
    #Store the prediction from the re-trained model
    nn.write_features(file_out_test, result)

    

    
