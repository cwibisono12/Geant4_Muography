#!/bin/bash 
#Generate Correlated Files from the Hits Files


directory_src=$1 #initial directory of the hits files
directory_final=$2 #final directory to store the correlated file
initial=$3 #Run number being proceseed
final=$4 #Final Run number being processed




if [[ "$initial" -ge 0 &&  "$initial" -lt 10000 ]];
	then
		for ((j=initial;j<=final;j++));
			do
			if [ -f "$directory_src/Run${j}_nt_ScintillatorHits.csv" ];
				then
					./main.py "$directory_src/Run${j}_nt_ScintillatorHits.csv" \
					"$directory_src/Run${j}_nt_PipeHits.csv" \
					"$directory_src/Run${j}_nt_ScalingHits.csv"\
					"$directory_final/corr_file_Run${j}_new.csv"
			else
				echo "File for Run $j not found in $directory_src"
			fi

		done

	fi



