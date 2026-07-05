Reproducible material for **DW0116:Depth-progressive 3D seismic velocity model building via 2D generative diffusion model - Shijun Cheng, Randy Harsuko, Mohammad H. Taufik, and Tariq Alkhalifah.**

[Click here](https://kaust.sharepoint.com/:b:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0116/DW0116_EAGE2026_Depth-progressive%203D%20seismic%20velocity%20model%20building%20via%202D%20generative%20diffusion%20model.pdf?csf=1&web=1&e=TNnWNb) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* **diffvmb3d**: python library containing routines for 3D velocity model building using 2D generative diffusion models;
* **dataset**: folder to store dataset;

## Supplementary files
To ensure reproducibility, we provide the the synthetic dataset for training and testing stages and our trainined model. 

* **Training and testing data set**
Download the training and testing data set [here](). Then, use `unzip` to extract the contents to `dataset/`.

* **Trained model**
Download our trained model [here](). Then, extract the contents to `/checkpoints/`.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate diffvmb3d
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Running code :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can run the training and inference code. 
For traning, you can directly run:
```
python train.py
```

For inference, you can use the test data we provide and directly run:
```
python sample.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `diffvmb3d/train.py` file.

## Acknowledgements
This implementation is motivated from the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open source code.

## Cite us 
DW0116 - Cheng et al. (2025) Depth-progressive 3D seismic velocity model building via 2D generative diffusion model.

