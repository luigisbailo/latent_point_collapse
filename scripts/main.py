import sys
import yaml
import argparse
import pickle
import importlib
import numbers
import os
import time
import datetime
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import defaultdict

import lpc_ib.networks as networks
from lpc_ib.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--results-dir', required=True)
parser.add_argument('--dataset-dir', required=True)
parser.add_argument('--model-name', required=True)
parser.add_argument('--loss-encoding', required=True)
parser.add_argument('--sample', required=False)
parser.add_argument('--lr', required=False)
parser.add_argument('--epochs', required=False)
parser.add_argument('--logging', required=False)
parser.add_argument('--dropout-penultimate', required=False)
parser.add_argument('--penultimate-nodes', required=False)
parser.add_argument('--encoding-metrics', required=False)
parser.add_argument('--store-penultimate', required=False)

def main():

    args = parser.parse_args()
    main_worker(args)

def main_worker ( args ):

    config_file = args.config
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    model_name = args.model_name
    lr = args.lr
    epochs = args.epochs
    logging = args.logging
    loss_encoding = args.loss_encoding
    dropout_penultimate = args.dropout_penultimate
    penultimate_nodes = args.penultimate_nodes
    encoding_metrics = args.encoding_metrics
    store_penultimate = args.store_penultimate
    sample = args.sample

    configs = parse_config(config_file)
    architecture = configs['architecture']
    training_hypers = configs['training']['hypers']
    name_dataset = configs['dataset']['name']

    # Overwrite learning rate if provided as argument
    if lr:
        if lr.startswith('.'):
            lr = '0'+lr
        training_hypers['lr'] = float(lr)

    # Overwrite epochs if provided as argument
    if epochs:
        training_hypers['epochs'] = int(epochs)

    # Overwrite logging if provided as argument
    if logging:
        training_hypers['logging'] = int(logging)


    # If loss_encoding is not provided, default to False
    loss_encoding = loss_encoding.lower() if loss_encoding else False
    if loss_encoding not in ['true', 'false']:
        print("Invalid value for loss_encoding. Defaulting to False.")
        loss_encoding = False
    else:
        loss_encoding = loss_encoding == 'true'
        
    # If encoding_metrics is not provided, default to False
    encoding_metrics = encoding_metrics.lower() if encoding_metrics else False
    if encoding_metrics not in ['true', 'false']:
        print("Invalid value for encoding_metrics. Defaulting to False.")
        encoding_metrics = False
    else:
        encoding_metrics = encoding_metrics == 'true'
        
    # If dropout is not provided, default to False
    dropout_penultimate = dropout_penultimate.lower() if dropout_penultimate else False
    if dropout_penultimate not in ['true', 'false']:
        print("Invalid value for dropout. Defaulting to False.")
        dropout_penultimate = False
    else:
        dropout_penultimate = dropout_penultimate == 'true'
    architecture['hypers']['dropout_penultimate'] = dropout_penultimate
    
    # If store_penultimate is not provided, default to False
    store_penultimate = store_penultimate.lower() if store_penultimate else False
    if store_penultimate not in ['true', 'false']:
        print("Invalid value for store_penultimate. Defaulting to False.")
        store_penultimate = False
    else:
        store_penultimate = store_penultimate == 'true'

    if penultimate_nodes == 'wide':
        penultimate_nodes_name = 'wide_'
        penultimate_nodes = architecture['hypers']['penultimate_nodes_wide']
    elif penultimate_nodes == 'narrow':
        penultimate_nodes_name = 'narrow_'        
        penultimate_nodes = architecture['hypers']['penultimate_nodes_narrow']
    else:
        penultimate_nodes_name = ''
        penultimate_nodes = architecture['hypers']['penultimate_nodes']

    transform = transforms.Compose([transforms.ToTensor()])
    torch_module = importlib.import_module("torchvision.datasets")

    if (name_dataset == 'SVHN'):
        torch_dataset = getattr(torch_module, name_dataset)
        trainset = torch_dataset(
            str(dataset_dir), split='train', download=True, transform=transform)
    elif (name_dataset == 'ImageNet'):
        torch_dataset = getattr(torch_module, name_dataset)
        trainset = torch_dataset(
            str(dataset_dir), split='train', transform=transform)
    else :
        torch_dataset = getattr(torch_module, name_dataset)
        trainset = torch_dataset(
            str(dataset_dir), train=True, download=True, transform=transform)
    
    gaussnoise_std=0.01
    trainset_mean, trainset_std = compute_mean_std(trainset)
    if (name_dataset == 'SVHN'):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trainset_mean, trainset_std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(trainset[0][0][0][0].shape[0], padding=4),
            AddGaussianNoise(mean=0, std=gaussnoise_std),
            Resize((32, 32)),  
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trainset_mean, trainset_std),
            Resize((32, 32)),  
        ])
    else:
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(trainset[0][0][0][0].shape[0], padding=4),    
        AddGaussianNoise(mean=0, std=gaussnoise_std),
       ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trainset_mean, trainset_std),
        ])
    
    if (name_dataset == 'SVHN'):
        trainset = torch_dataset(
        str(dataset_dir), split='train', download=True, transform=transform_train)
        testset = torch_dataset(
            str(dataset_dir), split='test', download=True, transform=transform_test)
    else:
        trainset = torch_dataset(
            str(dataset_dir), train=True, download=True, transform=transform_train)
        testset = torch_dataset(
            str(dataset_dir), train=False, download=True, transform=transform_test)

    input_dims = trainset[0][0].numel()
    if (name_dataset == 'SVHN'):
        num_classes = 10
    else:
        num_classes = len(set(trainset.classes))

    training_hypers = convert_bool(training_hypers)
    architecture['hypers'] = convert_bool(architecture['hypers'])
    
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))    
    world_size = torch.cuda.device_count()
    rank = dist.get_rank()

    file_name =  penultimate_nodes_name + model_name  
    if dropout_penultimate:
        file_name = 'dropout_' + file_name
    if sample:
        file_name = file_name + '_' + sample
    
    flag_file = f"{results_dir}/training_{file_name}.flag"
    checkpoint_file = f"{results_dir}/checkpoint_{file_name}.pth.tar"
    results_file = f"{results_dir}/{file_name}.pkl"
    if rank == 0:
        if os.path.exists(flag_file):
            print(f"Flag file {flag_file} exists. Another training job might be running. Exiting.")
            for _ in range(world_size):
                dist.send(torch.tensor([1]), dst=_)
            return
        else:
            with open(flag_file, 'w') as f:
                f.write('')
        print('Training ' + str(file_name))
        print('Learning rate: ', training_hypers['lr'])    
        print('World size: ', world_size)
        print('Penultimate nodes: ', penultimate_nodes)
    
        start_time = time.time()

    try:
        classifier = getattr(networks, architecture['backbone'])(
            model_name=model_name,
            architecture=architecture,
            num_classes=num_classes,
            penultimate_nodes = penultimate_nodes,
            input_dims=input_dims,
        )
        trainer = Trainer(
            network=classifier,
            architecture=architecture,
            trainset=trainset,
            testset=testset,
            trainset_mean=trainset_mean,
            trainset_std=trainset_std,
            training_hypers=training_hypers,
            model_name=model_name,
            encoding_metrics=encoding_metrics,
            store_penultimate=store_penultimate,
            verbose=True
            )
        results = trainer.fit(checkpoint_file, rank, world_size)
        print('rank: ', rank)

        if rank == 0:
            elapsed_time = time.time() - start_time
            print(f"Training time: {elapsed_time/60} minutes")

            results_file = os.path.join(results_dir, f'{file_name}.pkl')
            final_results = results
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'rb') as file:
                        results_checkpoint = pickle.load(file)
                    final_results = merge_results(results_checkpoint, results)
                except Exception as e:
                    print(f"Error loading results file: {e}")

            final_results['training_hypers'] = training_hypers
            final_results['architecture'] = architecture
            with open(results_file, 'wb') as file:
                pickle.dump(final_results, file)
    finally:
        if rank == 0:
            if os.path.exists(flag_file):
                os.remove(flag_file)

    dist.destroy_process_group()

    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def parse_config(config_file):
    """
    Parse the given config file and return the data.

    Args:
        config_file (str): The path to the config file.

    Returns:
        dict: The parsed data from the config file.

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there is an error parsing the config file.
    """
    try:
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def convert_bool(dictionary):
    """
    Converts string values 'true' and 'false' in the dictionary to boolean
    values.

    Args:
        dictionary (dict): The dictionary to be converted.

    Returns:
        dict: The dictionary with string values converted to boolean values.
    """
    for key, value in dictionary.items():
        if type(value) is str:
            if value.lower() == 'true':
                dictionary[key] = True
            elif value.lower() == 'false':
                dictionary[key] = False
    return dictionary

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        dataset: The dataset to compute the mean and standard deviation for.

    Returns:
        mean: The mean of the dataset.
        std: The standard deviation of the dataset.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(loader))[0].numpy()
    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))
    return mean, std

def merge_results(old_results, new_results):
    """
    Recursively merge new results into the old results.
    If the old key's value is False, it gets replaced by the new value.
    """
    if old_results is None:
        return new_results
    if new_results is None:
        return old_results
    
    if isinstance(old_results, dict) and isinstance(new_results, dict):
        for key, value in new_results.items():
            if key in old_results:
                if old_results[key] is False:
                    old_results[key] = value
                else:
                    old_results[key] = merge_results(old_results[key], value)
            else:
                old_results[key] = value
        return old_results
    elif isinstance(old_results, np.ndarray) and isinstance(new_results, np.ndarray):
        return np.concatenate((old_results, new_results))
    else:
        if old_results is False:
            return new_results
        return old_results
        
if __name__ == '__main__':
    
    main()
