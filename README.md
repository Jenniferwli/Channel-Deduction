# Channel Deduction:A New Learning Framework to Acquire Channel from Outdated Samples and Coarse Estimate
Zirui Chen, Zhaoyang Zhang, Zhaohui Yang, Chongwen Huang, Mérouane Debbah  
***
This repository is the official implementation of paper [Channel Deduction:A New Learning Framework to Acquire Channel from Outdated Samples and Coarse Estimate](https://ieeexplore.ieee.org/abstract/document/10845822).
## Overview of this work
To address the two key challenges in next-generation wireless systems—the high pilot overhead of channel estimation and the performance degradation from dynamic changes and error propagation in channel prediction—we propose a new framework named __Channel Deduction (CD)__. The core insight of this framework is the complementarity of estimation and prediction: the former uses current but limited information, while the latter, despite leveraging rich historical information, struggles to cope with present-time randomness and error accumulation. Channel Deduction is the first unified framework designed to synergistically fuse these two information sources.  

Specifically, Channel Deduction innovatively combines outdated channel samples from past time slots with a coarse estimate of the present channel obtained via very few pilots to acquire a complete and accurate channel representation. To achieve this, we designed Channel Deduction Networks (CDNets), which implement a deep fusion of time-space-frequency information through a specialized architecture: the networks utilize a CMixer module to process correlations in the space and frequency domains , and then use a time-domain interaction module to handle the channel's temporal evolution. Based on this, we propose two specific architectures: __the Recurrence-based RCDNet (using LSTM)__ and __the Attention-based ACDNet (using Transformer)__.  

Experimental results show that this method achieves high-quality channel acquisition while __reducing pilot overhead by up to 88.9%__. Furthermore, it maintains stable and continuous operation even under complex user mobility and error propagation, significantly outperforming traditional prediction methods. Channel Deduction marks an important step toward achieving robust and practical wireless AI.
## Main result
<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/fc5ecbef-7aa0-4979-b6d1-c954e63e8361" alt="Performance under a single scenario" style="width:95%;">
      <br>
      <em>Figure 1. NMSE of proposed CDNets and benchmarks under various estimated present channel sizes</em>
    </td>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/52251350-d7cc-479d-9081-72090b232284" alt="Performance of direct cross-scenario reuse" style="width:95%;">
      <br>
      <em>Figure 3. The trajectory of a user’s movement in ‘O1’ scenario (2024 time slots)</em>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/d52493b4-ff0c-4523-9702-187391a1466e" alt="Performance of multi-scenario joint learning and new scenario generalization" style="width:95%;">
      <br>
      <em>Figure 2. The generalization of CDNets on testing sets (NMSE) during training</em>
    </td>
    <td width="50%" align="center">
      <img src="https://github.com/user-attachments/assets/f0589064-8005-452b-b80f-c466cf16ca9a" alt="Performance comparisons between ALLoc and conventional new scenario tuning/training" style="width:95%;">
      <br>
      <em>Figure 4. NMSE between the acquired channel and the true channel during movement.</em>
    </td>
  </tr>

</table>

## Usages
### Dataset
This project is based on the __DeepMIMO 'O1' dataset__. To prepare the required data, please first download the original dataset from the following link:  

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
## Citation
If you find this work useful in your research, please consider citing us:  
```bibtex
@ARTICLE{10845822,
  author={Chen, Zirui and Zhang, Zhaoyang and Yang, Zhaohui and Huang, Chongwen and Debbah, Mérouane},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Channel Deduction: A New Learning Framework to Acquire Channel From Outdated Samples and Coarse Estimate}, 
  year={2025},
  volume={43},
  number={3},
  pages={944-958},
  keywords={Channel estimation;Estimation;OFDM;Correlation;Accuracy;Hands;Costs;Resistance;Mathematical models;Long short term memory;Channel acquisition;channel estimation;channel deduction;deep learning;massive MIMO;OFDM},
  doi={10.1109/JSAC.2025.3531576}}
