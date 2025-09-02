# Channel Deduction:A New Learning Framework to Acquire Channel from Outdated Samples and Coarse Estimate
Zirui Chen, Zhaoyang Zhang, Zhaohui Yang, Chongwen Huang, Mérouane Debbah
## Overview of this work

## Main result
<img width="2423" height="1126" alt="638201207681559266" src="https://github.com/user-attachments/assets/fc5ecbef-7aa0-4979-b6d1-c954e63e8361" />

## Usages
### Dataset
This project is based on the __DeepMIMO dataset__. To prepare the required data, please first download the original dataset from the following link:  

*Google Drive*: https://drive.google.com/drive/folders/1d5g5FczyaGndYtXRsLyiTQ1UMRYmNTQb?usp=drive_link  

Move the downloaded compressed file into the `/Data` directory and run the `decompress.sh` script from there to extract it.  

After the extraction is complete, execute the `pathchange.sh` script to replace the original absolute paths with current directory.  

Run the `data_generation.sh` script to process the raw DeepMIMO data and generate the final training and testing datasets required for this project.  
In `data_generation.sh`:  
* `data_division.py` partitions a large, grid-based spatial dataset into thousands of smaller, randomized blocks, then samples data points from within these blocks to create and save separate training and testing .npy files for a machine learning task.  
* `data_division_specifictest.py` generates two distinct test datasets from existing data collections：a __"mobile"__ dataset consists of completely random sequences and a __"static"__ dataset specifically constructed to create sequences with high correlation between the past and present data points.
### Training and Testing
The file structure of the repository is described in `directory_structure.txt.`  
* `input_4ant*4car/ACDNet/`: This directory contains the implementation of the ACDNet method proposed in the paper.  
* `input_4ant*4car/RCDNet/`: This directory contains the implementation of the RCDNet method proposed in the paper.  
* `input_4ant*4car/Estimation/`: This directory contains the implementation of the baseline method from the paper, which performs channel estimation using CMixer.  

Each method includes three core files: `model.py`, `train.py`, and `test.py`. To simplify experiments, a corresponding `.sh` script is provided for each method. We have also included pre-trained weights (`model.pth`), allowing you to run tests directly. If you wish to retrain the model, simply uncomment the training command within the `.sh` script.  

Additionally, the script files for ACDNet and RCDNet also contain an experiment to __test the continuous channel deduction service of CDNet for mobile users__. It first programmatically generates a smooth, irregular user trajectory (`plot_trace_smooth.py`) and then runs a simulation along this path (`test_trace_smooth.py`). The simulation compares the model's performance under __error propagation__ against __under the ideal case of no error propagation__, thereby validating the model's stability and robustness for practical application.  
### Note
Run `decompress.sh` inside the `/Data` folder to extract the dataset. The other scripts are executed inside the `/CD` folder (`pathchange.sh`, `data_generation.sh`, `ACDNet.sh`, `RCDNet.sh`, `Estimation.sh`).
