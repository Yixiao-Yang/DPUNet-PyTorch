#############################################################################
# DPUNet: Dynamic Proximal Unrolling Network for Compressive imaging [PyTorch version]

Yixiao Yang (yixiaoyang@bit.edu.cn), Ran Tao, Kaixuan Wei, Ying Fu

[[arXiv](https://arxiv.org/abs/2107.11007v1)]

## including codes of CS for natural image (BCS), CS for magnetic resonance imaging (CS-MRI) and CS for phase retrieval (CPR)

The code is built on **PyTorch** and tested on Ubuntu 16.04/18.04 and Windows 10 environment (Python3.x, PyTorch>=0.4) with 3080Ti GPU.

#############################################################################
Introduction 

A dynamic proximal unrolling network (dubbed DPUNet) was proposed, which can handle a variety of measurement matrices via one single model without retraining. 
Experimental results demonstrate that the proposed method can effectively handle multiple compressive imaging modalities under varying sampling ratios 
and noise levels via only one trained model, and outperform the state-of-the-art approaches.
#############################################################################
 Run test code
### Specific Compressive Imaging Task

#############################################################################
1. BCS Task:
Main train code: Train_DPUNet_BCS_noiseless_fixed.py, Train_DPUNet_BCS_noiseless_learned.py and Train_DPUNet_BCS_noisy_fixed.py
Main test code: TEST_DPUNet_BCS_noiseless_fixed.py, TEST_DPUNet_BCS_noiseless_learned.py and Test_DPUNet_BCS_noisy_fixed.py

1\ TEST_DPUNet_BCS_noiseless_fixed: Our method for block-based image compressive sensing task with fixed Gaussian sampling matrix under noiseless setting.
Note that it is required to change the parameters: sampling_ratio in the line 23.

2\ TEST_DPUNet_BCS_noiseless_learned: Our method for block-based image compressive sensing task with learned sampling matrix under noiseless setting.
Note that it is required to change the parameters: sampling_ratio in the line 22.

3\ TEST_DPUNet_BCS_noisy_fixed: Our method for block-based image compressive sensing task with fixed Gaussian sampling matrix under noisy setting.
Note that it is required to change the parameters: sampling_ratio and noisel_level in the line 23,24.

#############################################################################
2. CS-MRI Task:
Main train code: Train_DPUNet_CSMRI_noisy.py
Main test code: Test_DPUNet_CSMRI_noisy.py

Test_DPUNet_CSMRI_noisy: Our method for compressive sensing-MRI task.
Note that it is required to change the parameters: sampling_ratio and noisel_level in the line 21,22.

#############################################################################
3. CPR Task:
Main train code: Train_DPUNet_CPR_noisy.py 
Main test code: Test_DPUNet_CPR_noisy.py 

Test_DPUNet_CPR_noisy: Our method for compressive sensing phase retrieval task.
Note that it is required to change the parameters: sampling_ratio and noisel_level in the line 20,21.

4. MIX-alltasks:
Main train code: Train_MIX_DPUNet_noisy.py 
Main test code: Test_MIX_DPUNet_BCS.py, Test_MIX_DPUNet_CSMRI.py, Test_MIX_DPUNet_CPR.py  

Test_MIX_DPUNet_BCS: The extension version of our method to multitasks for image compressive sensing task.
Test_MIX_DPUNet_CSMRI: The extension version of our method to multitasks for compressive sensing MRI task.
Test_DPUNet_CPR_noisy: The extension version of our method to multitasks for compressive sensing phase retrieval task.
Note that it is required to change the parameters: sampling_ratio and noisel_level.

#############################################################################
 Run Train code
You need to download the Training Dataset and move it into the ./*/data (* denotes BCS/CSMRI/CPR/MIX_alltasks)
Baidu drive with the link: 
https://pan.baidu.com/s/1HhMSIvUUq-MycgaiGzofug?pwd=dpun 
code：dpun 
#############################################################################
BCS training dataset: Training_Data.mat                                                for BCS and MIX_alltasks
CS-MRI training dataset: Training_BrainImages_256x256_100.mat          for CSMRI and MIX_alltasks
CPR training dataset: Training_Data_64_160000.mat                              for CPR and MIX_alltasks

#############################################################################

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```

```
## Acknowledgements