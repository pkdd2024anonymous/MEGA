import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import  DataLoader
import dgl
import pandas as pd
import os
from sklearn.decomposition import PCA
import argparse
import yaml
import torch.nn as nn
from sklearn.metrics import accuracy_score
import sys
import torch.optim as optim
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import torch
import src.data
import src.evaluator
from src.utils import get_presaved_embeddings_and_tensors


# turn off warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)



def check_config_validity(exp_dict):
    assert len(exp_dict['tensor_paths']) == len(exp_dict['tensor_labels'])
     
def accuracy(output, target):
    predictions = output.argmax(dim=1, keepdim=True)
    correct = predictions.eq(target.view_as(predictions)).sum().item()
    return correct / target.size(0)



def aggregate_tensors(tensor, aggr_strategy='sum'):
    # aggregate tensors of [num_tasks, num_nodes, embd_dim] to [num_nodes, embd_dim]
    if aggr_strategy == 'sum':
        return torch.sum(tensor, dim=0)
    elif aggr_strategy == 'mean':
        return torch.mean(tensor, dim=0)
    elif aggr_strategy == 'max':
        return torch.max(tensor, dim=0)[0]
    elif aggr_strategy == 'concat':
        return tensor.view(tensor.shape[1], -1)
    elif aggr_strategy == 'pca':
        np_tensor = tensor.cpu().numpy()
        output_dim = np_tensor.shape[2]
        # turn [num_tasks, num_nodes, embd_dim] into [num_nodes, num_tasks * embd_dim]
        np_tensor = np_tensor.transpose(1, 0, 2).reshape(np_tensor.shape[1], -1)
        pca = PCA(n_components=output_dim)
        data_pca = pca.fit_transform(np_tensor)
        return torch.from_numpy(data_pca).to(device)
    else:
        raise ValueError('Invalid aggregation strategy')
    
def get_processed_tensors(tensor, tensor_label, aggr_strategies):
    proccessed_tensors, proccessed_tensor_labels = [], []

    if len(tensor.shape) == 3:
        print("Aggregating tensors for {}".format(tensor_label))
        for agg_str in aggr_strategies:
            agg_tensor = aggregate_tensors(tensor, agg_str)
            proccessed_tensors.append(agg_tensor)
            proccessed_tensor_labels.append(agg_str + " " + tensor_label)
    elif len(tensor.shape) == 2:
        print("(2-d tensor) No need to aggregate tensors for {}".format(tensor_label))
        proccessed_tensors.append(tensor)
        proccessed_tensor_labels.append(tensor_label)
    else:
        raise ValueError('Invalid tensor shape')
    
    return proccessed_tensors, proccessed_tensor_labels
    
    return proccessed_tensors

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



 

def run_link_prediction(exp_dict):
    check_config_validity(exp_dict)
    print(exp_dict)

    
    export_dir = exp_dict['export_dir']
    os.makedirs(export_dir, exist_ok=True)
    
    target_mode = 'link_prediction'
    report_df_path = os.path.join(export_dir, f'{target_mode}_report_df.csv')
    if not os.path.exists(report_df_path) or exp_dict['overwrite']:
        report_df = pd.DataFrame()
    else:
        print("Reading report df from {}".format(report_df_path))
        print("Appending to existing report df...")
        report_df = pd.read_csv(report_df_path)

    dataset = exp_dict['dataset_name']
    tvt_edges_file = f'links/{dataset}_tvtEdges.pkl'

    for tensor_path, tensor_label in zip(exp_dict['tensor_paths'], exp_dict['tensor_labels']):
        print(f'Running downstream tasks for {tensor_label}')
        original_emb_tensor = torch.load(tensor_path, map_location=device).to(device)
        proccessed_tensors, proccessed_tensor_labels = get_processed_tensors(original_emb_tensor, tensor_label, exp_dict['aggr_strategies'])
        
        for data, label in zip(proccessed_tensors, proccessed_tensor_labels):
            print(f'{target_mode}: Running {target_mode} for {label}, shape of tensor: {data.shape}, datset: {exp_dict["dataset_name"]}')
            auc, auc_std, hits20, hits20_std = src.evaluator.fit_link_predictor(data, tvt_edges_file, device, exp_dict['batch_size'], exp_dict['neg_rate'], es_metric='auc', epochs=exp_dict['epochs'], patience=exp_dict['patience'], repeat=exp_dict['repeats'])

            result_dict = {
                'auc_mean': auc,
                'auc_std': auc_std,
                'hits20_mean': hits20,
                'hits20_std': hits20_std,
                'tensor_label': label,
                'export_time': pd.Timestamp.now()
            }
            df_to_append = pd.DataFrame(result_dict, index=[0])
            report_df = pd.concat([report_df, df_to_append], ignore_index=True)

        report_df.to_csv(report_df_path, index=False)

        # delete all tensors
        del original_emb_tensor
        del proccessed_tensors
        del data



            
        
        

            

import traceback


if __name__ == '__main__':
    EXPERIMENT_PATH = 'scripts/SSL_training_2345_link'
    datasets = ['arxiv','co_computer', 'co_photo', 'pubmed',  'wiki_cs',  'co_cs', 'chameleon', 'squirrel','actor', ] #'products'
    with open('downstream_experiments/experiments/common_settings_link.yaml', 'r') as f:
        common_settings = yaml.safe_load(f)


    for dataset in datasets:
        #dataset_root_tensor_path = os.path.join('scripts_2/SSL_training_2345', dataset)
        dataset_root_tensor_path = os.path.join(EXPERIMENT_PATH, dataset)
        tensors_paths, tensor_labels = get_presaved_embeddings_and_tensors(dataset_root_tensor_path)

        exp_dict = {
            'dataset_name': dataset,
            'export_dir': f'downstream_experiments/exports_2345/link/{dataset}',
            'tensor_paths': tensors_paths,
            'tensor_labels': tensor_labels,

        }
        exp_dict.update(common_settings)
        # pretty print the exp_dict
        print("=====================================")
        print("Running downstream tasks for dataset: ", dataset)
        print("=====================================")
        print(yaml.dump(exp_dict))
        print("=====================================")

        run_link_prediction(exp_dict)
        print("Done with dataset: ", dataset)
        print("=====================================")
