## ✅ To-Do List for MotionDissolve Release

- ✅ Release the final test result
- ✅ Release the processing code
- ✅ Release the second finetune process
- [  ] Update the final model weights (The model weights are pending internal review and will be uploaded shortly.)
- [  ] Release the source code
- [  ] Release the first-stage training process

## Installation

Create a conda environment & Install requirments 

```shell
conda create -n motiondissolve python==3.10.16
conda activate motiondissolve
pip install -r requirements.txt

```

If you encounter an error while installing Flash Attention, please [**manually download**](https://github.com/Dao-AILab/flash-attention/releases) the installation package based on your Python version, CUDA version, and Torch version, and install it using `pip install flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.

## Data Processing

You can leverage processing_event on the original event data to obtain a fixed frame event input.

## Test

```shell
python ./AA_1_Final_submit/finetune_2stage/output3_test_guass.py
```

- running ./AA_1_Final_submit/finetune_2stage/output3_test_guass.py to obtain the final output ./AA_1_Final_submit/finetune_2stage/results/bin

```shell
python ./AA_1_Final_submit/finetune_2stage/output3_test_image_time.py
```

- running ./AA_1_Final_submit/finetune_2stage/output3_test_image_time.py to obtain an inference time of our methods.

## Visualization

```shell
python ./AA_1_Final_submit/finetune_2stage/bin2rgb.py
```

- running ./AA_1_Final_submit/finetune_2stage/bin2rgb.py to obtain visualization of our output


## Acknowledgement

Thanks to the inspiration and codes from [AHDINet](https://github.com/wyang-vis/AHDINet), thanks for their generous open source.
