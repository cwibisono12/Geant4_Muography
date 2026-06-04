#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
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

    dim = len(X_train)
    for i in range(dim):
        print(i, X_train[i], y_train[i])
    
    layer = (500,100,100)
    nn.retrain_model(X_train, y_train, file_model)
    
    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    result = nn.predict_outcome_from_file(X_test, file_model)

    score = nn.get_model_score_from_file(X_test, y_test, file_model)

    print("score:",score)

    nn.store_score_result('../corr_files/retrain_test_score.csv', layer, 2000, score)

    
    #Store the prediction from the re-trained model
    nn.write_features(file_out_test, result)

    

    
