# IFPGen
This is a repository of our paper, "Iterative interaction fingerprints-guided multiple-objective molecular generation" [link]()
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
```
Python scripts/train_IFP.py configs/training.yml --device {DEVICE} --logdir {LOGDIR} --index {INDEX} --processed {PROCESSED}`
```

```
Arguments:  
   --device     Computing device (default = "cuda") 
   --logdir     Directory where the training results are saved (str)
   --index      Dataset index file (str)
   --processed  Processed dataset path (str)
```
### Trained model checkpoint
[Link]()
## Sampling 
### preprocessing
For sampling ligands via IFPGen, the pre-sampling file `input_json.json` needs to be prepared before sampling:
```
Python scripts/preprocess_IFPGen.py --pdb_path {PDB_PATH} --pdb_metal_path {PDB_METAL_PATH} --ref_ligand_path {REF_LIGAND_PATH} --multiple_pdb {MULTIPLE_PDB} --output {OUTPUT}
```

```
Arguments:  
   --pdb_path           Path to PDB file (optional for single input)
   --pdb_metal_path     Path to PDB metal file (optional for single input)
   --ref_ligand_path    Path to reference ligand SDF file (optional for single input)
   --multiple_pdb       Path to list file (.pkl or .json) containing multiple inputs
   --output             Output directory for IFP files and JSON config
```
Output `input_json.json` containing information about one or more targets, depending on whether `multiple_pdb` is provided. Users can manually adjust the IFP.csv and `fix_ifp` to achieve the interaction customization function.

The content of `input_json.json` is as follows. 
```
[
  {
    "pocket_idx": 0,
    "pdb_path": 'pocket.pdb',
    "ref_ligand_sdf": 'ref_ligand.sdf',
    "pdb_metal_path": 'pocket_metal.pdb',
    "ref_ifp": 'IFP.csv',
    "fix_ifp": [[x,y]]
   },
         ......
]
```
### sampling (non-iterative/iterative)
You can enable IFPGen's iterative optimization function by setting `max_iteration` to be greater than 1, and you can determine the number of objectives and weights to optimize by setting `score_components` (current one-to-many considerations vina score, qed, sa, posebuster four indicators)
```
Python scripts/IFPGen.py --output_dir {OUTPUT_DIR} --device {DEVICE} --ckpt_path {CKPT_PATH} --config {CONFIG} --input_json {INPUT_JSON} --batch_size {BATCH_SIZE} --sample_size {SAMPLE_SIZE} --seed {SEED} --max_iterations {MAX_ITERATIONS} --top_k {TOP_K} --overwrite {OVERWRITE} --score_components {SCORE_COMPONENTS}
```

```
Arguments:  
   --output_dir       Output directory for sampling results
   --max_iterations   Maximum number of iterations (set to 1 for no iteration optimization)        
   --sample_size      Number of molecules to sample
   --top_k            Number of top molecules to select
   --device           Computing device
   --ckpt_path        Path to model checkpoint
   --config           Config file for model sampling
   --input_json       Path to input data JSON
   --batch_size       Batch size for optimization
   --seed             Random seed
   --overwrite        Overwrite existing output files
   --score_components Comma-separated component:weight pairs for combined_score, e.g., 'vina:1.5,qed:1.0,sa:1.0,posebuster:1.0' (default: vina:1.5,qed:1.0,posebuster:1.0)
```
## Ligands sampled from CrossDocked2020 test set
We generated 100 ligands for each pocket of the CrossDocked2020 benchmark set via the non-iterative and iterative methods of IFPGen respectively, following the splitting method proposed by [Luo et al](https://arxiv.org/pdf/2203.10446).

You can download the generated molecules from [Zenodo]() (two methods, 10,000 molecules each).

## Citation
```
lol
```
