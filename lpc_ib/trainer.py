import importlib
import time 
import random
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from lpc_ib.metrics import (
    get_entropy,
    get_coeff_var,
    get_collapse_metrics,
    get_binarity_metrics,
    get_mahalanobis_score,
    get_odin_score,
)
from lpc_ib.deepfool import deepfool


class Trainer():
    """
    A class that represents a trainer for a neural network classifier.

    Parameters:
    - device (torch.device): The device to use for training.
    - network (torch.nn.Module): The neural network model to train.
    - trainset (torch.utils.data.Dataset): The training dataset.
    - testset (torch.utils.data.Dataset): The testing dataset.
    - training_hypers (dict): A dictionary containing hyperparameters for training.
    - model (str): The type of model being trained \
        ('ib', 'lin_pen', 'nonlin_pen', 'no_pen').
    - encoding_metrics (bool): Whether to calculate encoding metrics during training.
    - store_penultimate (bool): Whether to store penultimate layer activations after training.
    - verbose (bool): Whether to print training progress.

    Methods:
    - fit(): Trains the neural network model.
    - eval(dataset): Evaluates the neural network model on a given dataset.

    Returns:
    - res_dict_stack (dict): A dictionary containing the stacked training results.
    """

    def __init__(self, network, architecture, trainset, testset, trainset_mean, trainset_std, training_hypers, model_name,
                 encoding_metrics=True, store_penultimate=False, verbose=True):
        self.network = network
        self.architecture = architecture
        self.trainset = trainset
        self.testset = testset
        self.trainset_mean = trainset_mean
        self.trainset_std = trainset_std
        self.training_hypers = training_hypers
        self.model_name = model_name
        self.encoding_metrics = encoding_metrics
        self.store_penultimate = store_penultimate
        self.verbose = verbose
        
            
    def fit(self, checkpoint_file, rank, world_size):
        """
        Trains the neural network model.

        Returns:
        - res_dict_stack (dict): A dictionary containing the stacked training results.
        """
        
        torch.cuda.set_device(rank)
        model = self.network.to(rank)

        if world_size>1:
            model_ddp = DDP(model, device_ids=[rank])
            excluded_params = set(model_ddp.module.output_layer.parameters())
        else:
            model_ddp = model
            excluded_params = set(model_ddp.output_layer.parameters())

        other_params = [param for name, param in model_ddp.named_parameters() if param not in excluded_params]
        excluded_params = list(excluded_params)  
        params_to_update = [
            {'params': other_params, 'lr': self.training_hypers['lr']},
            {'params': excluded_params, 'lr': self.training_hypers['lr']}
        ]
        torch_optim_module = importlib.import_module("torch.optim")
        self.opt = getattr(torch_optim_module, self.training_hypers['optimizer'])(
            params_to_update,
            weight_decay=self.training_hypers['weight_decay']
        )  
        self.scheduler = StepLR(
            self.opt, step_size=self.training_hypers['lr_scheduler_step_size'], gamma=self.training_hypers['lr_scheduler_gamma']
        )
           
        start_epoch = 0
        gamma = self.training_hypers['gamma']
        gamma_max = 10**self.training_hypers['gamma_max_exp']
        checkpoint_loaded = False
        if (rank==0):
            try:
                checkpoint = load_checkpoint(checkpoint_file)
                model_ddp.load_state_dict(checkpoint['state_dict'])
                self.opt.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] 
                gamma = checkpoint['gamma']
                checkpoint_loaded = True
                print(f"Resuming from checkpoint at epoch {start_epoch}")
            except FileNotFoundError:
                print("No checkpoint found, starting training from scratch")

        checkpoint_loaded = torch.tensor(checkpoint_loaded).to(rank)
        dist.broadcast(checkpoint_loaded, src=0)
        if checkpoint_loaded.item():
            state_dict_list = [model_ddp.state_dict(), self.opt.state_dict(),  self.scheduler.state_dict(), start_epoch, gamma]
            dist.broadcast_object_list(state_dict_list, src=0)

            model_ddp.load_state_dict(state_dict_list[0])
            self.opt.load_state_dict(state_dict_list[1])
            self.scheduler.load_state_dict(state_dict_list[2])
            start_epoch = state_dict_list[3]
            gamma = state_dict_list[4]

        self.train_sampler = DistributedSampler(self.trainset, num_replicas=world_size, rank=rank)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.training_hypers['batch_size'],
            sampler=self.train_sampler
        )
        
        converged = False
        convergence_thres = self.training_hypers['convergence_thres']

        res_list = []
        for epoch in range(start_epoch+1, start_epoch+self.training_hypers['epochs']+1):
            self.train_sampler.set_epoch(epoch)  
            model_ddp.train()

            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(rank)
                y_batch = y_batch.to(rank)
                x_output_batch, x_penultimate_batch = model_ddp(x_batch)

                self.opt.zero_grad()

                loss_classification = nn.CrossEntropyLoss(reduction='mean')(
                    x_output_batch, y_batch).to(rank)
                loss = loss_classification

                if self.model_name == 'ib':
                    loss_encoding = nn.functional.mse_loss(
                        x_penultimate_batch,
                        torch.zeros(x_penultimate_batch.shape).to(rank),
                        reduction='mean').to(rank)
                    loss = loss + loss_encoding*gamma
                
                loss.backward()

                self.opt.step()           
            
            
            if epoch % self.training_hypers['gamma_scheduler_step'] == 0 and gamma < gamma_max:
                gamma = gamma * self.training_hypers['gamma_scheduler_factor']
            
            if epoch > self.training_hypers['lr_scheduler_start']:
                self.scheduler.step()        
                self.opt.param_groups[1]['lr'] = self.training_hypers['lr']

                
            last_epoch = epoch == self.training_hypers['epochs'] 
            logging_epoch = epoch % self.training_hypers['logging'] == 0

            if logging_epoch or last_epoch:

                res_epoch = {}                
                eval_trainloader = torch.utils.data.DataLoader(
                    self.trainset,
                    batch_size=4*self.training_hypers['batch_size'],
                    sampler=self.train_sampler
                )
                eval_train = self.eval(eval_trainloader, rank)
                
                gathered_eval_train = gather_dict_outputs_ddp(eval_train, rank, world_size)
                        
                if rank == 0:
                    eval_testloader = torch.utils.data.DataLoader(
                        self.testset,
                        batch_size=4*self.training_hypers['batch_size']
                    )
                    eval_test = self.eval(eval_testloader, rank)

                    res_epoch['epochs'] = epoch                    
                    res_epoch['accuracy_train'] = (gathered_eval_train['y_predicted'] == gathered_eval_train['y_label']).mean()
                    res_epoch['accuracy_test'] = (eval_test['y_predicted'] == eval_test['y_label']).mean()
                    if self.verbose:
                        print('Epoch', epoch)
                        print('gamma', gamma)
                        print(
                            'Accuracy train: ',
                            np.around(res_epoch['accuracy_train'], 4),
                            "\tAccuracy test:",
                            np.around(res_epoch['accuracy_test'], 4)
                        )

                    start_time = time.time()
                    res_epoch['collapse_train'] = get_collapse_metrics(gathered_eval_train)

                    print('collapse test done')
                    elapsed_time = time.time() - start_time
                    print(f"Elapsed time to compute metric collapse train: {elapsed_time/60} minutes" )
         
                    res_epoch['collapse_test'] = get_collapse_metrics(eval_test)

                    if res_epoch['accuracy_train'] > convergence_thres and converged==False :
                        converged = True
                        print('converged!', epoch)
                        convergence_epoch = epoch
                        accuracy_test_converged = res_epoch['accuracy_test']

                    if self.model_name == 'ib' or self.model_name == 'lin_pen':
                        start_time = time.time()
                        res_epoch['binarity_train'] = get_binarity_metrics(gathered_eval_train)
                        res_epoch['binarity_test'] = get_binarity_metrics(eval_test)
                        elapsed_time = time.time() - start_time
                        print(f"Elapsed time to compute binarity metrics: {elapsed_time/60} minutes" )

                    if last_epoch:
                        
                        batch_size_deepfool = 1000
                        loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size_deepfool, shuffle=True)
                        images, labels = next(iter(loader))
                        images = images.to(rank)
                        labels = labels.to(rank)

                        start_time = time.time()
                        perturbation_list = []
                        for image in images:
                            r_tot, loop_i, label, k_i, pert_image = deepfool(
                                image,
                                self.network
                            )
                            perturbation_list.append(
                                np.linalg.norm(r_tot) /
                                np.linalg.norm(image.cpu().numpy())
                            )

                        deepfool_score_tpt = np.mean(perturbation_list)
                        print(f"Elapsed time to compute DeepFool score: {elapsed_time/60} minutes" )
                 
                        mahalanobis_score_tpt = get_mahalanobis_score(gathered_eval_train, eval_test)
                        odin_score_tpt = get_odin_score(gathered_eval_train, eval_test, [], T=100)
                        entropy_train_tpt = get_entropy(gathered_eval_train, k=100)
                        entropy_test_tpt = get_entropy(eval_test, k=100)
                        coeff_var_train_tpt = get_coeff_var(gathered_eval_train)
                        coeff_var_test_tpt = get_coeff_var(eval_test)
                        
                        if self.store_penultimate:
                            penultimate_train = gathered_eval_train['x_penultimate']
                            penultimate_test = eval_test['x_penultimate']
                
                res_list.append(res_epoch)

        res_dict_stack = {}                
        if rank == 0:
            for key in res_list[0].keys():
                if isinstance (res_list[0][key], dict):            
                    res_dict_stack[key] = {} 
                    for key2 in res_list[0][key].keys():
                        if isinstance (res_list[0][key][key2], dict):            
                            res_dict_stack[key][key2] = {}
                            for key3 in res_list[0][key][key2].keys():
                                res_dict_stack[key][key2][key3] = np.vstack(
                                    [res_epoch[key][key2][key3] for res_epoch in res_list ]
                                )
                        else:                
                            res_dict_stack[key][key2] = np.vstack(
                                [res_epoch[key][key2] for res_epoch in res_list ]
                            )                    
                else:                
                    res_dict_stack[key] = np.vstack([res_epoch[key] for res_epoch in res_list])

            if converged:
                res_dict_stack['convergence_epoch'] = convergence_epoch
                res_dict_stack['accuracy_test_converged'] = accuracy_test_converged
            else:
                res_dict_stack['accuracy_test_converged'] = False
                res_dict_stack['convergence_epoch'] = False

            save_checkpoint({
                'epoch': epoch,
                'gamma': gamma,
                'state_dict': model_ddp.state_dict(),
                'optimizer': self.opt.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, checkpoint_file)
                
            if self.encoding_metrics:
                res_dict_stack['mahalanobis_score_tpt'] = mahalanobis_score_tpt
                res_dict_stack['odin_score_tpt'] = odin_score_tpt
                res_dict_stack['entropy_train_tpt'] = entropy_train_tpt 
                res_dict_stack['entropy_test_tpt'] = entropy_test_tpt 
                res_dict_stack['coeff_var_train_tpt'] = coeff_var_train_tpt 
                res_dict_stack['coeff_var_test_tpt'] = coeff_var_test_tpt 
                res_dict_stack['deepfool_score_tpt'] = deepfool_score_tpt

            if self.store_penultimate:
                res_dict_stack['penultimate_train'] = penultimate_train
                res_dict_stack['penultimate_test'] = penultimate_test

        return res_dict_stack
                

    def eval(self, loader, device):
        """
        Evaluates the neural network model on a given dataset.

        Parameters:
        - dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.

        Returns:
        - evaluations (dict): A dictionary containing the evaluation results.
        """

        evaluations = {}
        self.network.eval()

        with torch.no_grad():

            x_output = []
            y_label = []
            x_penultimate = []
            for x_batch, y_label_batch in loader:
                x_batch = x_batch.to(device)
                y_label_batch = y_label_batch.to(device)
                x_output_batch, x_penultimate_batch = self.network(x_batch)
                y_label.append(y_label_batch)
                x_output.append(x_output_batch)
                x_penultimate.append(x_penultimate_batch)
            x_output = torch.cat(x_output)
            x_penultimate = torch.cat(x_penultimate)
            y_predicted = torch.argmax(torch.softmax(x_output, dim=-1), dim=1)
            y_label = torch.cat(y_label)

            x_output = x_output.cpu().numpy()
            x_penultimate = x_penultimate.cpu().numpy()
            y_predicted = y_predicted.cpu().numpy()
            y_label = y_label.cpu().numpy()
            evaluations['x_output'] = x_output
            evaluations['x_penultimate'] = x_penultimate
            evaluations['y_predicted'] = y_predicted
            evaluations['y_label'] = y_label

        return evaluations

def save_checkpoint(state, filename):
    torch.save(state, filename)

    
def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    return checkpoint
                
    
def gather_dict_outputs_ddp(local_eval_train, rank, world_size):
                
    gathered_eval_train = {key: [] for key in local_eval_train.keys()}

    for key, local_data in local_eval_train.items():

        local_data_tensor = torch.tensor(local_data).to(rank)
        gathered_data = [torch.zeros_like(local_data_tensor) for _ in range(world_size)]
        dist.gather(local_data_tensor, gathered_data if rank == 0 else [], dst=0)
        
        if rank == 0:
            gathered_data = torch.cat(gathered_data, dim=0)
            gathered_eval_train[key] = gathered_data.cpu().numpy()

    return gathered_eval_train if dist.get_rank() == 0 else None
            
    
