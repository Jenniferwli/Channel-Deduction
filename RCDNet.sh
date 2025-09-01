#!/bin/bash
  
cd ./input_4ant*4car/RCDNet && \
#echo "RCDNet Model Training"
#python train.py && \
echo "RCDNet Testing"
python test.py && \
echo "RCDNet continuously serve a mobile user Testing"
python plot_trace_smooth.py && \
python test_trace_smooth.py

echo "RCDNet All scripts executed successfully."

