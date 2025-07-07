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

## Data Processing

You can leverage processing_event.py on the original event data to obtain a fixed frame event input.

## Test

```shell
python output3_test_guass.py
```

- running output3_test_guass.py to obtain the final output ./AA_1_Final_submit/finetune_2stage/results/bin

```shell
python output3_test_image_time.py
```

- running output3_test_image_time.py to obtain an inference time of our methods.

## Visualization

```shell
python bin2rgb.py
```

- running bin2rgb.py to obtain visualization of our output


## Acknowledgement

Thanks to the inspiration and codes from [AHDINet](https://github.com/wyang-vis/AHDINet), thanks for their generous open source.
