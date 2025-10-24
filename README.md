# IFPGen
This is a repository of our paper, "Iterative interaction fingerprints-guided multiple-objective molecular generation" (_)[link]
<img width="2319" height="1171" alt="overview_v2" src="https://github.com/user-attachments/assets/549e5a9e-f054-44b4-9519-ec621f2e3df3" />
we devised a novel 3D molecular generation and optimization framework which integrates conditional diffusion model and iterative optimization strategy to generate molecules with specific interactions and perform well on multiple objects. 

## Install packages
You can install the required packages by running the following commands:

`conda env create -n IFPGen -f environment.yml`

## Data processing
The data used for training / evaluating the model are organized in the [data]() Google Drive folder.

| Dataset      | Link |
| ----------- | ----------- |
| Crossdocked2020 |[https://bits.csb.pitt.edu/files/crossdock2020/](https://bits.csb.pitt.edu/files/crossdock2020/)|
| PDBbind   | [https://www.pdbbind-plus.org.cn/](https://www.pdbbind-plus.org.cn/) |
| MeDBA   |[https://medba.ddtmlab.org/](https://medba.ddtmlab.org/)|
## Training IFPGen
### Training from scratch
`Python scripts/train_diffusion_ifp.py configs/training.yml --device {DEVICE} --logdir {LOGDIR}`
```
Arguments:  
   --device    Computing device (default = "cuda") 
   --logdir    Directory where the training results are saved (str)
```
### Trained model checkpoint
[Link]()
## Sampling 
