#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import nn

    '''
    Below is an example how to use module above to predict the outcome over the test data.
    '''

    file_test = sys.argv[1]
    file_model = sys.argv[2]
    file_out_test = sys.argv[3]
    option = int(sys.argv[4])
    epoch = int(sys.argv[5])

    
    layer = (500,500,500)
    
    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    #Dimensionality Reduction:
    X_test_new = nn.select_features(X_test)

    result = nn.predict_outcome_from_file(X_test_new, file_model)

    score = nn.get_model_score_from_file(X_test_new, y_test, file_model)

    print("score:",score)

    nn.store_score_result('../../../corr_files/retrain_test_score_reduce_dim.csv', file_test, layer, epoch, score, file_model)

    
    #Store the prediction from the re-trained model:
    nn.write_features(file_out_test, result)

    

    
