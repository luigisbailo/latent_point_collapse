import argparse
import os
import glob 
import pickle
import sys
import torch 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', required=True)
parser.add_argument('--output-dir', required=True)

args = parser.parse_args()

results_dir = args.results_dir
output_dir = args.output_dir

models = ['ib', 'wide_ib', 'narrow_ib', 'no_pen', 'dropout_no_pen', 'lin_pen', 'dropout_lin_pen', 'nonlin_pen']

results = {}

for model in models:
    best_res_list = []    
    for entry in os.listdir(results_dir):

        if (os.path.isdir(results_dir + '/' + entry) and entry != '.ipynb_checkpoints'):
            max_acc = 0
            best_res = None
            file_pattern = f'{results_dir}{"/"}{entry}{"/"}{model}*{"pkl"}'
            files = glob.glob(file_pattern)

            for file in files:
                print(file)
                with open(file, 'rb') as file:
                    res = pickle.load(file)
                    acc = res['accuracy_test'][-1][0]
                    print('acc',acc)
                    if acc > max_acc:
                        best_res = res
                        max_acc = acc
            best_res_list.append(best_res)
    
    res_dict = {}
    for key in best_res_list[0].keys():
        if isinstance (best_res_list[0][key], dict):
            res_dict[key] = {}
            for key2 in best_res_list[0][key].keys():
                res_dict[key][key2] = np.hstack([res[key][key2] for res in best_res_list ])
        else:                
            res_dict[key] = np.hstack([res[key] for res in best_res_list ])
        
    results[model] = res_dict
    
with open (output_dir + '/best_results.pkl', 'wb') as file:
    pickle.dump(results, file)
