# OSCC-PathologyImageDataset

## Multi-OSCC Datasets 

### Dataset Download
The url is coming soon.

### Dataset splitation
Following the random seed 42, we split the dataset into training, validation, and test sets with a ratio of 70:15:15.
In `json` file: `./Data/split_seed=2024.json`

### Images labels
We save the images labels in `json` file: `./Data/all_metadata.json`


## Environment
Follow the instructions below to create a conda environment with the required dependencies.
```
conda env create -f environment.yml
conda activate Multi-OSCCPI
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
```


## Stain Normalization Visualization
[Stain normalization samples](./Data/visualize_diff_stain_method.png)


## Cite this work
If you use this dataset, please cite the following paper:
```
This will be set when the paper is published.
```