# Channel Deduction:A New Learning Framework to Acquire Channel from Outdated Samples and Coarse Estimate
Zirui Chen, Zhaoyang Zhang, Zhaohui Yang, Chongwen Huang, Mérouane Debbah
## Overview of this work

## Main result

## Usages
### Dataset
This project is based on the DeepMIMO dataset. To prepare the required data, please first download the original dataset from the following link:  

*Google Drive*: https://drive.google.com/drive/folders/1d5g5FczyaGndYtXRsLyiTQ1UMRYmNTQb?usp=drive_link  

Move the downloaded compressed file into the `/Data` directory and run the `decompress.sh` script from there to extract it.  

After the extraction is complete, execute the `pathchange.sh` script to replace the original absolute paths with current directory.  

Run the `data_generation.sh` script to process the raw DeepMIMO data and generate the final training and testing datasets required for this project.  
In `data_generation.sh`:  
* `data_division.py` partitions a large, grid-based spatial dataset into thousands of smaller, randomized blocks, then samples data points from within these blocks to create and save separate training and testing .npy files for a machine learning task.  
* `data_division_specifictest.py` generates two distinct test datasets from existing data collections：a "mobile" dataset consists of completely random sequences and a "static" dataset specifically constructed to create sequences with high correlation between the past and present data points.
### Training and Testing
The file structure of the repository is described in `directory_structure.txt.`  
* `input_4ant*4car/ACDNet/`: This directory contains the implementation of the ACDNet method proposed in the paper.  
* `input_4ant*4car/RCDNet/`: This directory contains the implementation of the RCDNet method proposed in the paper.  
* `input_4ant*4car/Estimation/`: This directory contains the implementation of the baseline method from the paper, which performs channel estimation using CMixer.  

Each method includes three core files: `model.py`, `train.py`, and `test.py`. To simplify experiments, a corresponding `.sh` script is provided for each method. We have also included pre-trained weights (`model.pth`), allowing you to run tests directly. If you wish to retrain the model, simply uncomment the training command within the `.sh` script.  

