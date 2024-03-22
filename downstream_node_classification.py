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
from src.utils import get_presaved_embeddings_and_tensors
import src.data

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensor[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


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

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    total_loss = 0
    total_acc = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_acc += accuracy_score(target.cpu().numpy(), predicted.cpu().numpy())
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    return avg_loss, avg_acc

def evaluate(model, criterion, test_loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_acc += accuracy_score(target.cpu().numpy(), predicted.cpu().numpy())
    avg_loss = total_loss / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    return avg_loss, avg_acc



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

def get_target_data(dataset, target='label'):
    assert target in ['label', 'node_assignment']
    g = src.data.load_data(dataset, 'pretrain_labels', False, 'none', split = 'random', hetero_graph_path = 'hetero_graphs')
    if target == 'label':

        targets = g.ndata['label'].long().to(device)
        if(dataset in ['arxiv', 'products']):
            # turn into one dimensional
            targets = targets.squeeze()
        node_ids = [i for i in range(g.number_of_nodes())]
    elif target == 'node_assignment':
        targets = g.ndata['node_assignment'].long().to(device)
        node_ids = [i for i in range(g.number_of_nodes())]

    if(dataset == 'arxiv'):
        print("---------")
        print("g.ndata['label'].long().to(device) shape", g.ndata['label'].long().to(device).shape)
        print(g.ndata['label'].long().to(device))
        print("---------")
        print("g.ndata['node_assignment'].long().to(device) shape", g.ndata['node_assignment'].long().to(device).shape)
        print(g.ndata['node_assignment'].long().to(device))
        print("---------")
    return targets, node_ids

'''
def read_target_data(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=0, names=["node_id", "feature", "label"], delimiter='\t')
    # Access the "label" column
    labels = df["label"].values

    df = pd.DataFrame({'label': labels}, index=range(len(labels)))

    # To set the index name to 'node_id'
    df.index.name = 'node_id'

    # Display the DataFrame
    print(df.head())

    # count number of classes 
    num_classes = len(set(df['label']))

    print("Successfully loaded data from {}".format(file_path))
    print('Number of classes:', num_classes)
    print("Labels included:", set(df['label']))

    targets =  df['label'].values
    node_ids = df.index.tolist()

    targets = torch.tensor(targets, dtype=torch.long).to(device)
    return targets, node_ids

'''

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def run_downstream_tasks_for_tensor(data, targets_and_ids, exp_dict):
    results_list = []

    input_dim = data.shape[1]

    target = targets_and_ids[0]
    node_ids = targets_and_ids[1]
    num_classes = len(np.unique(target.cpu().numpy()))
    for seed in exp_dict['seeds']:
        seed_everything(seed)



        # Split the data
        data_cpu = data.cpu().numpy()
        target_cpu = target.cpu().numpy()

        # print number of unique in traget_cpu
        print("Number of unique labels in target_cpu: ", len(np.unique(target_cpu)))

        # Find those that have at most 1 sample and remove
        unique, counts = np.unique(target_cpu, return_counts=True)
        to_remove = unique[counts == 1]
        # remove from node_ids
        node_ids = [i for i in node_ids if target_cpu[i] not in to_remove]
        print("Number of unique in target_cpu that have at most 1 sample: ", len(to_remove))
        print("Number of each unique in target_cpu that have at most 1 sample: ", to_remove)
        #remov from data_cpu and target_cpu
        print("Before removing, data_cpu shape: {}, target_cpu shape: {}".format(data_cpu.shape, target_cpu.shape))
        # Create a mask for all elements NOT to be removed
        mask = ~np.isin(target_cpu.flatten(), to_remove)  # Flatten to make mask 1D

        # Apply the mask to filter out rows in both data_cpu and target_cpu
        data_cpu = data_cpu[mask]
        target_cpu = target_cpu[mask]
        print("After removing, data_cpu shape: {}, target_cpu shape: {}".format(data_cpu.shape, target_cpu.shape))

        X_train_ids, X_test_ids, y_train, y_test = train_test_split(node_ids, target_cpu, test_size=0.2, random_state=seed, stratify=target_cpu)

        train_data, train_target = data[X_train_ids], target[X_train_ids]
        test_data, test_target = data[X_test_ids], target[X_test_ids]
        train_dataset = CustomDataset(train_data, train_target)
        test_dataset = CustomDataset(test_data, test_target)


        param_grid = exp_dict['training_param_grid']

        best_params = None
        best_val_loss = float('inf')


        if exp_dict['kfold'] == 'off':
            for params in ParameterGrid(param_grid):
                print("Running training (Kfold off) for seed {},  params: {}".format(seed, params))

                # Model, criterion, and optimizer for this parameter set
                model = SimpleNN(input_size=input_dim, hidden_size=params['hidden_size'], num_classes=num_classes)
                model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

                # Training and evaluation without k-fold
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

                epochs = params['epoch']
                best_fold_epoch = 0
                best_fold_train_loss = float('inf')
                best_fold_val_loss = float('inf')
                best_fold_train_acc = 0
                best_fold_val_acc = 0

                pbar = tqdm(range(epochs), position=0, leave=True)

                
                for epoch in pbar:
                    model.train()
                    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader)

                    if train_loss < best_fold_train_loss:
                        best_fold_train_loss = train_loss
                        best_fold_train_acc = train_acc
                        best_fold_epoch = epoch

                        best_test_loss, best_test_acc = evaluate(model, criterion, test_loader)

                        pbar.set_postfix(OrderedDict(
                            best_epoch=best_fold_epoch,
                            best_train_loss=best_fold_train_loss,
                            best_train_acc=best_fold_train_acc,
                            best_test_loss=best_test_loss,
                            best_test_acc=best_test_acc,
                        ))
                
                results_list.append({'seed': seed, 'info': 'validation', 'fold': 'off', 'epochs': epochs, 'learning_rate': params['learning_rate'], 
                                    'batch_size': params['batch_size'], 'hidden_size': params['hidden_size'],
                                    'best_fold_epoch': best_fold_epoch, 'best_fold_train_loss': best_fold_train_loss,
                                        'best_fold_train_acc': best_fold_train_acc, 'best_test_loss': best_test_loss, 
                                        'best_test_acc': best_test_acc})

            print("For params: {} \n\t test_loss: {}".format(params, best_test_loss))
            if best_test_loss < best_val_loss:
                best_val_loss = best_test_loss
                best_params = params

        else:


            skf = StratifiedKFold(n_splits=exp_dict['kfold'])  # 5-fold stratified cross-validation

            for params in ParameterGrid(param_grid):
                kfold_val_losses = []

                print("Running cross fold for seed {},  params: {}".format(seed, params))
                current_fold = 0
                train_data_cpu = train_data.cpu().numpy()
                train_target_cpu = train_target.cpu().numpy()
                for train_idx, val_idx in skf.split(train_data_cpu, train_target_cpu):
                    # Creating data loaders for the current fold
                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_subsampler)
                    val_loader = DataLoader(train_dataset, batch_size=len(val_subsampler), sampler=val_subsampler)

                    # Model, criterion, and optimizer
                    
                    model = SimpleNN(input_size=input_dim, hidden_size=params['hidden_size'], num_classes=num_classes)
                    print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))
                    model.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

                    # Training and evaluation
                    epochs = params['epoch']
                    best_fold_epoch = 0
                    best_fold_train_loss = float('inf')
                    best_fold_val_loss = float('inf')
                    best_fold_train_acc = 0
                    best_fold_val_acc = 0

                    pbar = tqdm(range(epochs), position=0, leave=True)

                    for epoch in pbar:
                        model.train()
                        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader)

                        if train_loss < best_fold_train_loss:
                            best_fold_train_loss = train_loss
                            best_fold_train_acc = train_acc
                            best_fold_epoch = epoch

                            best_fold_val_loss, best_fold_val_acc = evaluate(model, criterion, val_loader)

                            pbar.set_postfix(OrderedDict(
                                best_epoch=best_fold_epoch,
                                best_train_loss=best_fold_train_loss,
                                best_train_acc=best_fold_train_acc,
                                best_val_loss=best_fold_val_loss,
                                best_val_acc=best_fold_val_acc,
                            ))
                    
                    kfold_val_losses.append(best_fold_val_loss)
                    results_list.append({'seed': seed, 'info': 'validation', 'fold': current_fold, 'epochs': epochs, 'learning_rate': params['learning_rate'], 
                                        'batch_size': params['batch_size'], 'hidden_size': params['hidden_size'],
                                        'best_fold_epoch': best_fold_epoch, 'best_fold_train_loss': best_fold_train_loss,
                                            'best_fold_train_acc': best_fold_train_acc, 'best_fold_val_loss': best_fold_val_loss, 
                                            'best_fold_val_acc': best_fold_val_acc})
                    current_fold += 1

                avg_val_loss = np.mean(kfold_val_losses)
                print("For params: {} \n\t avg_val_loss: {}".format(params, avg_val_loss))
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = params

        print("Best params: {}".format(best_params))
        print("Best validation loss: {}".format(best_val_loss))
        # Training on the whole training set with the best params
        # if number of parametersets are more than 1 should run
        if len(ParameterGrid(param_grid)) > 1:
            print("Running one last training with the best params and evaluating on the test set for seed {}".format(seed))

            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            model = SimpleNN(input_size=input_dim, hidden_size=best_params['hidden_size'], num_classes=num_classes)
            print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))
            model = model.to(device)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

            epochs = best_params['epoch']
            best_epoch = 0
            best_train_loss = float('inf')
            best_test_loss = float('inf')
            best_train_acc = 0
            best_test_acc = 0

            pbar = tqdm(range(epochs), position=0, leave=True)

            for epoch in pbar:
                model.train()
                train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader)

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_acc = train_acc
                    best_epoch = epoch

                    best_test_loss, best_test_acc = evaluate(model, criterion, test_loader)

                    pbar.set_postfix(OrderedDict(
                        best_epoch=best_epoch,
                        best_train_loss=best_train_loss,
                        best_train_acc=best_train_acc,
                        best_test_loss=best_test_loss,
                        best_test_acc=best_test_acc,
                    ))

            print("+++++ For BEST params: {} \n\t best_epoch: {} \n\t best_train_loss: {} \n\t best_train_acc: {} \n\t best_test_loss: {} \n\t best_test_acc: {}".format(
                best_params, best_epoch, best_train_loss, best_train_acc, best_test_loss, best_test_acc))
            
            results_list.append({'seed': seed, 'info': 'test', 'epochs': epochs, 'learning_rate': best_params['learning_rate'], 
                                        'batch_size': best_params['batch_size'], 'hidden_size': best_params['hidden_size'],
                                        'best_epoch': best_epoch, 'best_train_loss': best_train_loss,
                                            'best_train_acc': best_train_acc, 'best_test_loss': best_test_loss, 
                                            'best_test_acc': best_test_acc})
        
        else:
            print("Just one paramterset so no need to run one last training with the best params and evaluating on the test set for seed {}".format(seed))
    # delete all tensors
    del data
    del train_data
    del test_data
    del train_target
    del test_target
    del train_dataset
    del test_dataset
    del model

    return results_list



 

def run_downstream_tasks(exp_dict):
    check_config_validity(exp_dict)

    export_dir = exp_dict['export_dir']
    os.makedirs(export_dir, exist_ok=True)
    
    for target_mode in exp_dict['targets']:
        # check if report df already exist, read it, otherwise new df
        report_df_path = os.path.join(export_dir, f'{target_mode}_report_df.csv')
        if not os.path.exists(report_df_path) or exp_dict['overwrite']:
            report_df = pd.DataFrame()
        else:
            print("Reading report df from {}".format(report_df_path))
            print("Appending to existing report df...")
            report_df = pd.read_csv(report_df_path)

        for tensor_path, tensor_label in zip(exp_dict['tensor_paths'], exp_dict['tensor_labels']):
            print(f'Running downstream tasks for {tensor_label}')
            original_emb_tensor = torch.load(tensor_path, map_location=device).to(device)
            proccessed_tensors, proccessed_tensor_labels = get_processed_tensors(original_emb_tensor, tensor_label, exp_dict['aggr_strategies'])
            
            results_list = []
            if(target_mode == 'node_clustering'):
                raise ValueError('Node clustering not supported')
            else:
                # add different label support
                targets_and_ids = get_target_data(exp_dict['dataset_name'], target_mode)
                for data, label in zip(proccessed_tensors, proccessed_tensor_labels):
                    print(f'{target_mode}: Running {target_mode} for {label}, shape of tensor: {data.shape}, datset: {exp_dict["dataset_name"]}')
                    results_list = run_downstream_tasks_for_tensor(data, targets_and_ids, exp_dict)

                    # for each dict in results_list, add add label and then append to report_df
                    export_time = pd.Timestamp.now()
                    for result_dict in results_list:
                        result_dict['tensor_label'] = label
                        result_dict['export_time'] = export_time
                        df_to_append = pd.DataFrame(result_dict, index=[0])
                        report_df = pd.concat([report_df, df_to_append], ignore_index=True)

            report_df.to_csv(report_df_path, index=False)

            # delete all tensors
            del original_emb_tensor
            del proccessed_tensors
            del targets_and_ids



            
        
        

            

import traceback


if __name__ == '__main__':

    """
    EXPERIMENT_PATH = 'scripts_2/SSL_training_2345_ssnc'
    datasets = ['co_computer', 'co_photo', 'pubmed', 'squirrel', 'wiki_cs', 'arxiv', 'co_cs', 'chameleon', 'actor', ]#'products'
    with open('downstream_experiments/experiments/common_settings_ssnc.yaml', 'r') as f:
        common_settings = yaml.safe_load(f)


    for dataset in datasets:
        #dataset_root_tensor_path = os.path.join('scripts_2/SSL_training_2345', dataset)
        dataset_root_tensor_path = os.path.join(EXPERIMENT_PATH, dataset)
        tensors_paths, tensor_labels = get_presaved_embeddings_and_tensors(dataset_root_tensor_path)

        exp_dict = {
            'dataset_name': dataset,
            'export_dir': f'downstream_experiments/exports_2345/ssnc/{dataset}',
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

        run_downstream_tasks(exp_dict)
        print("Done with dataset: ", dataset)
        print("=====================================")
    """

    ## ABLATIONS 
    EXPERIMENT_PATH = 'scripts_2/ABL_SSL_training_2345_ssnc'
    datasets = ['chameleon',]#'products'
    with open('downstream_experiments/experiments/common_settings_ssnc.yaml', 'r') as f:
        common_settings = yaml.safe_load(f)


    for dataset in datasets:
        #dataset_root_tensor_path = os.path.join('scripts_2/SSL_training_2345', dataset)
        dataset_root_tensor_path = os.path.join(EXPERIMENT_PATH, dataset)
        tensors_paths, tensor_labels = get_presaved_embeddings_and_tensors(dataset_root_tensor_path)

        exp_dict = {
            'dataset_name': dataset,
            'export_dir': f'downstream_experiments/exports_ABL_2345/ssnc/{dataset}',
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

        run_downstream_tasks(exp_dict)
        print("Done with dataset: ", dataset)
        print("=====================================")