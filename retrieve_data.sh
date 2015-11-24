#!/bin/bash

#Usage: sh retrieve_data.sh Project Obs

LTA_folder="/data/Daniele/B2217+47/LTA/"

cd $LTA_folder

#if [[ $1 == *"ulsars"* ]]; then

  screen -dm lta-retrieve.py --query -p $1 $2

#else

#  if [ `ls $1.csv | wc -w` = 0 ]; then 
  
#    lta-query.py -p $1
  
#  fi

#  screen -d -m lta-retrieve.py --csvfile=$1.csv $2 

#fi


