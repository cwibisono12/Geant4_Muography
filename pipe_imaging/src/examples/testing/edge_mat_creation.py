#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    import utilities
    from threading import Thread

    '''
    Generate 2D matrices for edge detection
    '''

    directory_in = sys.argv[1] #directory input where the merged prediction files are 
    dim_y = int(sys.argv[2]) #y dimension for matrix creation
    dim_x = int(sys.argv[3]) #x dimension for matrix creation
    directory_out = sys.argv[4] #directory output to store matrix file
    option = int(sys.argv[5]) #data retrieval mode when running the simulation
    run_mode = int(sys.argv[6]) #data retrieval mode when running the model for training and testing
    epoch = int(sys.argv[7]) #iteration number used when running the model.
    f_log = sys.argv[8]

    #Get the list of file name inside directory:
    file_list = utilities.retrieve_files_in_directory(directory_in)

    #Only process the files with the following:
    pattern = str(option)+'_'+str(run_mode)+'_'+str(epoch)

    for name in file_list:

        #Get the file name inside directory input:
        temp_fname = utilities.retrieve_merged_file_name(str(name))
        
        if pattern in temp_fname:
            print(f"Begin Processing {temp_fname} files")
            
            #Create 2D matrices for edge detection analysis:
            mat_1, mat_2 = utilities.detect_edge(name, dimy = dim_y, dimx = dim_x)
    
            #Get the 2D matrices file name to store the 2D matrices:
            fmat_1 = directory_out + temp_fname +'_2D_int.mat'
            fmat_2 = directory_out + temp_fname +'_2D_fdr.mat'

            t1 = Thread(target = utilities.matwrite, args=(fmat_1,), kwargs = {'dimy':dim_y, 'dimx':dim_x, 'arr':mat_1, 'overwrite':1})
            t2 = Thread(target = utilities.matwrite, args=(fmat_2,), kwargs = {'dimy':dim_y, 'dimx':dim_x, 'arr':mat_2, 'overwrite':1})
    
            t1.start()
            t2.start()

            t1.join()
            t2.join()
            
            #Create log:
            utilities.store_log_2D_mat_edge(f_log, temp_fname, temp_fname+'_2D_int', temp_fname+'_2D_fdr')

