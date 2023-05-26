# H-NeXt
## The next step towards roto-translation invariant networks
## Prerequisites
* Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Trained using RockyLinux and NVIDIA A100
* Create conda environment 
``
conda env create -f environment.yml
``
## Training
* Training CLI in [training.py](./h-next/training.py)
  * Specify ID of GPU by CUDA_VISIBLE_DEVICES
  * To choose size of batch, data dir, for zero padding pad
    * `--datamodule_hparams "{'batch_size': 63, 'data_dir' : '/tmp', 'pad':0}"`
  * For more details see [training.py](./h-next/training.py) 
* Parameters are optimized for [Wandb Sweeps](https://wandb.ai/site/sweeps), see examples in: [data/sweeps](./data/sweeps). 
### 0. Activate Conda Env
``` 
conda activate h-next
```
### MNIST Models
Each architecture design is represented by the best model of 10 runs and how to train them is listed bellow:

#### 1. H-Nets
```bash
CUDA_VISIBLE_DEVICES=0 python training.py 
```
#### 2. **UP** 
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetBackbone"
```
#### 3. **UP + MASK**
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetBackbone" --backbone_hparams "{'circular_masking':True}"
```
### CIFAR-10 Models 
#### 1. **UP + MASK**
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetBackbone" --backbone_hparams "{'maximum_order': 1, 'circular_masking':True, 'in_channels':3}" --datamodule_name "cifar10-rot-test"
```
#### 2. **UP + MASK + HUGE**
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetBackbone" --backbone_hparams "{'maximum_order': 2, 'circular_masking':True, 'in_channels':3, 'nf1':32, 'nf2':64, 'nf3':128}" --datamodule_name "cifar10-rot-test"
```
#### 3. **UP + MASK + WIDE**
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetWideBackbone" --classnet_name "ZernikeProtypePooling" --datamodule_name "cifar10-rot-test"
```
#### 4. **UP + MASK + ATT**
```bash
CUDA_VISIBLE_DEVICES=0 python training.py  --backbone_name "UpscaleHnetWideBackbone" --backbone_hparams "{'model_str' : 'B-8-MP,B-16' }" --classnet_name "TransformerPooling" --datamodule_name "cifar10-rot-test"
```
## Testing
* Artifacts of models are listed in [data/models](./data/models), and divided according to datasets.
* How to load and test model see [testing.ipynb](./h-next/testing.ipynb).
* For SWN-GCN evaluation same models as for mnist-rot-test and cifar10-rot-test were used, thus their training datasets are equal. 

## Datasets
When using our datasets, they will be downloaded automatically see: [custom_datasets.py](./h-next/custom_datasets.py)
Direct links: 
* [mnist-rot-test](https://owncloud.cesnet.cz/index.php/s/q2BYzg8Uzcc8O4g/download)
* [cifar10-rot-test](https://owncloud.cesnet.cz/index.php/s/Denv319G7GwulEv/download) 

# Troubleshooting
## libcublasLt.so.11
* https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no
```bash
export LD_LIBRARY_PATH=~/miniconda3/lib:"$LD_LIBRARY_PATH"
```