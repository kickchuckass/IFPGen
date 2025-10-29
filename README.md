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
`Python scripts/train_IFP.py configs/training.yml --device {DEVICE} --logdir {LOGDIR} --index {INDEX} --processed {PROCESSED}`
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

`Python scripts/preprocess_IFPGen.py --pdb_path {PDB_PATH} --pdb_metal_path {PDB_METAL_PATH} --ref_ligand_path {REF_LIGAND_PATH} --list_file {LIST_FILE} --output {OUTPUT}`
```
Arguments:  
   --pdb_path           Path to PDB file (optional for single input)
   --pdb_metal_path     Path to PDB metal file (optional for single input)
   --ref_ligand_path    Path to reference ligand SDF file (optional for single input)
   --list_file          Path to list file (.pkl or .json) containing multiple inputs
   --output             Output directory for IFP files and JSON config
```
### sampling (non-iterative/iterative)
`Python scripts/IFPGen.py --output_dir {OUTPUT_DIR} --device {DEVICE} --ckpt_path {CKPT_PATH} --config {CONFIG} --input_json {INPUT_JSON} --batch_size {BATCH_SIZE} --sample_size {SAMPLE_SIZE} --seed {SEED} --max_iterations {MAX_ITERATIONS} --top_k {TOP_K} --overwrite {OVERWRITE} --score_components {SCORE_COMPONENTS}`
```
--output_dir       Output directory for sampling results
--max_iterations   Maximum number of iterations (**set to 1 for no iteration optimization**) 
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
