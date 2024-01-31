# Deep Heterogeneous Contrastive Hyper-Graph Learning for In-the-Wild Context-Aware Human Activity Recognition

## Source Code for IMWUT 2023 Issue 4 paper
Deep Heterogeneous Contrastive Hyper-Graph Learning for In-the-Wild Context-Aware Human Activity Recognition. <br>
-- Authors: [Guanyi Mou](https://scholar.google.com/citations?user=OdJ_YZMAAAAJ&hl=en)[^1], [Wen Ge](https://scholar.google.com/citations?user=h8P5Z3UAAAAJ&hl=en)[^1], [Emmanuel O. Agu](https://web.cs.wpi.edu/~emmanuel/?_gl=1*sn0www*_ga*MTI5NzQzMzAxMi4xNzA2Mjk1OTQ0*_ga_RE35PKQB7J*MTcwNjcxMzI5Mi4xMC4wLjE3MDY3MTMyOTIuNjAuMC4w), and [Kyumin Lee](https://web.cs.wpi.edu/~kmlee/)

 [^1]: Equal contribution.

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

## Citations
If you find our work useful, please consider cite our work
```
@article{ge2024deep,
  title={Deep Heterogeneous Contrastive Hyper-Graph Learning for In-the-Wild Context-Aware Human Activity Recognition},
  author={Ge, Wen and Mou, Guanyi and Agu, Emmanuel O and Lee, Kyumin},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={7},
  number={4},
  pages={1--23},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

Contact [wge@wpi.edu](wge@wpi.edu) for any further information.
