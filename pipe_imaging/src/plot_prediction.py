#!/usr/bin/env python3


if __name__ == "__main__":
    import sys
    import nn

    f_test = sys.argv[1]
    f_predict = sys.argv[2]

    #nn.plot_prediction(fin)
    nn.plot_test_and_predict(f_test, f_predict)
