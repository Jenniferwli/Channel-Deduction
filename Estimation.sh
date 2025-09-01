#!/bin/bash
  
cd ./input_4ant*4car/Estimation && \
#echo "Estimation Model Training"
#python train.py && \
echo "Estimation Testing"
python test.py


echo "Estimation All scripts executed successfully."

