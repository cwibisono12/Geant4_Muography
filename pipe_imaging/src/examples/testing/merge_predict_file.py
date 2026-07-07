#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities

    directory_in = sys.argv[1] #directory input where the prediction files are
    option = int(sys.argv[2]) #data retrieval mode when running the simulation (eg. 3 for retrieving all hits on the objects)
    run_mode = int(sys.argv[3]) #data retrieval mode when running the model for training and testing (1 to 9)
    initial = int(sys.argv[4]) #begin run number to be merged
    final = int(sys.argv[5]) #final run number to be merged
    epoch = int(sys.argv[6]) #iteration number when running the model for training
    fout = sys.argv[7] #file to store the merged data
    flog = sys.argv[8] #log file to keep the record

    #Get the list of file name inside directory:
    file_list = utilities.retrieve_files_in_directory(directory_in)

    for name in file_list:
        temp = str(name)
        for k in range(initial, final+1, 1):
            pattern = 'Run'+str(k)+'_'+str(option)+'_'+str(run_mode)+'_'+str(epoch)
            
            #Merge file that meets condition above:
            if pattern in temp:
                #Merge temp into fout:
                utilities.merge_prediction_file(temp, fout, flog)

