# imwut_test

## Source Code for IMWUT 2023 Issue 4 paper
Deep Heterogeneous Contrastive Hyper-Graph Learning for In-the-Wild Context-Aware Human Activity Recognition

More content details to come in the following days

## What are in the repo
- bash script for running the code ``trainer.sh``
- folder ``code`` containing all python code
- folder ``data`` containing sampled data (not the full data)

We showcase an example for runnning the given code on a sampled data slice. Extrasensory is a public dataset and researchers should be able to download and process the full original source dataset (we do not own this dataset). For more details, please refer to the original paper.

## How to run the code
- Make sure the required packages are installed with compatible versions. We are aware that torch_geometrics are sensitive to different versions of pytorch
- Unzip folders under data (even the sampled users with sampled example files are large)
- Modify the trainer.sh file hyper-parameter settings
- run the script with ``bash trainer.sh``
- Check printline logs in ``log`` folder and the results in ``output`` folder

Contact wge@wpi.edu for any further information.