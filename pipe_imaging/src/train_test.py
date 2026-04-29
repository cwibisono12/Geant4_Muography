#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_out = sys.argv[3]

    X_train, y_train = nn.get_features(file_train)
    print(X_train[2], y_train[2])
    model_param = nn.get_parameters(X_train, y_train)

    X_test, y_test = nn.get_features(file_test)
    
    result = nn.predict_outcome(X_test, model_param)

    nn.write_features(file_out, result)
    nn.plot_results(result)
    
