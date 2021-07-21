#!/usr/bin/env python3
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import argparse
import importlib
import collections
import itertools

import utils
from datasets.data_loader import get_loaders
from networks.network import LLL_Net

if torch.cuda.is_available():
    device = 'cuda'

def iter_task_models(network_cls, taskcla, result_file):
    path = os.path.join(result_file, 'models')
    models_dict = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            task_num = int(''.join([n for n in name if n.isdigit()]))
            models_dict[task_num] = name
        
        models_dict = collections.OrderedDict(sorted(models_dict.items()))
        for task_num, name in models_dict.items():
            print(name)
            # Load the network
            net = getattr(importlib.import_module(name='networks'), network_cls)
            model = net(pretrained=False)
            model = LLL_Net(model, remove_existing_head=True)
            # Add heads
            for task in range(task_num + 1):
                model.add_head(taskcla[task][1])
            model.set_state_dict(torch.load(os.path.join(path, name)))
            model = model.eval()
            model.to(device)
            yield model
            
def get_accumulative_accuracies(test_loaders, taskcla, result_file, network_cls='resnet32'):
    """ Confusion matrix with progressively more classes considered """
    iter_model = iter_task_models(network_cls, taskcla, result_file)
    accuracies = np.zeros((len(taskcla), len(taskcla)))
    classes_so_far = 0.
    for task_model, model in enumerate(iter_model):
        for task_eval in range(0, task_model+1):
            full_test_loader = itertools.chain.from_iterable(test_loaders[:task_eval+1])
            with torch.no_grad():
                totals = 0.
                correct = 0.
                logits_mask = np.arange(sum([taskcla[i][1] for i in range(0, task_eval+1)]))
                for inputs, targets in full_test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = torch.cat(model(inputs), dim=1)
                    outputs = outputs[:, logits_mask]
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).double().sum()
                    totals += len(targets)
            accuracies[task_model, task_eval] = correct/totals
            
    return accuracies

def extract_cumulative_accuracy(path):
    """ Computes cumulative accuracy, the models need to be stored inside result_dir/models """

    def find(path, key):
        """ returns the first file containing key in its name """
        for root, dirs, files in os.walk(path):
            for name in files:
                if key in name:
                    return os.path.join(path, name)
                
    argfile = find(path, 'args')
    print(f'Using config {argfile}')

    with open(argfile, 'r') as f:
        contents = f.read()
        contents = contents.replace('false', 'False')
        contents = contents.replace('true', 'True')
        contents = contents.replace('null', 'None')
        print(contents)
        argdict = eval(contents)

    print(argdict)
    args = argparse.Namespace(**argdict)

    # Load the loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, 
    args.nc_first_task, args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Compute accuracies
    accuracies = get_accumulative_accuracies(tst_loader, taskcla, path, args.network)
    
    return accuracies

def extract_forgetting(accs):
    """ Compute forgetting matrices """
    num_tasks = len(accs)
    forgetting = np.zeros_like(accs)
    for task_eval in range(num_tasks):
        for task_train in range(task_eval + 1, num_tasks):
            forgetting[task_train, task_eval] = np.max(accs[:task_train + 1, task_eval] - accs[task_train, task_eval])
    return forgetting

def main(args):
    """ Computes both cumulative forgetting and accuracies, store them in the results path """
    
    accuracies = extract_cumulative_accuracy(args.load)
    forgettings = extract_forgetting(accuracies)
    
    path = os.path.join(args.load, 'cumulative_accs')
    np.save(path, accuracies)
    path = os.path.join(args.load, 'cumulative_forg')
    np.save(path, forgettings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load', type=str, help='result dir for which to compute the metrics (must be generated with --save-models option)')
    args = parser.parse_args()
    main(args)
