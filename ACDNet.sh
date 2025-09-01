#!/bin/bash

cd ./input_4ant*4car/ACDNet && \
#echo "ACDNet Model Training"
#python train.py && \
echo "ACDNet Testing"
python test.py && \
echo "ACDNet continuously serve a mobile user Testing"
python plot_trace_smooth.py && \
python test_trace_smooth.py

echo "ACDNet All scripts executed successfully."

