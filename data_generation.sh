#!/bin/bash

# 顺次执行Python脚本
echo "Data preprocessing"
python ./datavision/data_division.py && \
python ./datavision/data_division_specifictest.py && \

echo "Data generate successfully."
