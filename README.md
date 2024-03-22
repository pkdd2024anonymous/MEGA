# MEGA: Multi-Encoder GNN Architecture for stronger task collaboration and generalization submitted to PKDD 2024


This anonymized repository contains the code for the paper "MEGA: Multi-Encoder GNN Architecture for stronger task collaboration and generalization submitted to PKDD 2024". The code is based on the PyTorch Geometric library and is used to reproduce the results of the paper. The code is organized as follows:


1. Install the required packages:

```
* DGL 0.9.0
* PyTorch 1.12.0
* ogb 1.3.4
```


2. Generate datasets with these scripts:

```
python hetero_graph_gen.py
python link_gen.py
```

3. Pretrain models on all datasets, for both link and node tasks (You can also modify the script to run a selection of experiments):

```
cd scripts
bash universal_script.sh
```


4. Evaluate pre-trained models on all datasets for node tasks (i.e., node classification and partition classification):

```
python downstream_node_classification.py 
```

You can modify the contents of `downstream_experiments/experiments/common_settings_ssnc.yaml`  to change experimental setup. 

5. Evaluate pre-trained models on all datasets for link tasks (i.e., link prediction):

```
python downstream_link_prediction.py 
```

You can modify the contents of `downstream_experiments/experiments/common_settings_link.yaml`  to change experimental setup.


6. For ablations, you can pretrain models using the following script:

```
cd scripts
ssnc_chameleon_ablations.sh
```


7. To review and aggregate the results, you can use the following notebook:

```
results_analysis.ipynb
```