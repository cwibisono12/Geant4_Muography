#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities

    fin = sys.argv[1] #file input to be merged
    fout = sys.argv[2] #file to append fin to merged file
    flog = sys.argv[3] #log file to keep the record

    #merge fin into fout:
    utilities.merge_prediction_file(fin, fout, flog)
