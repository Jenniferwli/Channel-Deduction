#!/bin/bash

# Execute Python scripts sequentially
echo "Data preprocessing"
python ./datavision/data_division.py && \
python ./datavision/data_division_specifictest.py && \

echo "Data generate successfully."
