#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    '''
    Below is an example how to use module above to asses the score of the model over the test data.
    '''

    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_score = sys.argv[3]
    option = int(sys.argv[4])

    X_train, y_train = nn.get_features_append(file_train, arg_number = option)

    dim = len(X_train)
    for i in range(dim):
        print(i, X_train[i], y_train[i])
   
    
    #1) Iteration
    first_trial = (500,500,500,100)
    model_param = nn.get_parameters(X_train, y_train, layer = first_trial, num_iter = 2000)

    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    result = nn.predict_outcome(X_test, model_param)

    nn.write_features('../corr_files/corr_file_Run2_3_3_500500500100_test.csv', result)
    score = nn.get_model_score(X_test, y_test, model_param)

    nn.store_score_result(file_score,file_test, first_trial, 2000, score) 

    del X_test
    del y_test

    #2) Iteration
    second_trial = (500,500,500,500)
    
    model_param = nn.get_parameters(X_train, y_train, layer = second_trial, num_iter = 2000)

    X_test, y_test = nn.get_features_append(file_test, arg_number = option)
    
    result = nn.predict_outcome(X_test, model_param)
    nn.write_features('../corr_files/corr_file_Run2_3_3_500500500500_test.csv', result)

    score = nn.get_model_score(X_test, y_test, model_param)

    nn.store_score_result(file_score, file_test, second_trial, 2000, score) 


