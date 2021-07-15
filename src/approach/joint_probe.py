import torch
import os
import numpy as np
import itertools
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from .joint import JointDataset

""" Special approach, that starts from a learned sequence of models, and learns the heads on all the dataset """


class Appr(Inc_Learning_Appr):
    """Class implementing the approach from Workshop CL CVPR 2021"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, load_dir=None, bias=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset)

        self.trn_datasets = []
        self.val_datasets = []
        self.load_dir = load_dir
        self.bias = bias


    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--load_dir', default=None,
                            help='Directory from which to fetch the models')
        parser.add_argument('--bias', action='store_true', default=False,
                            help='only tune the bias')
                                   
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self._train_parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _train_parameters(self):
        """Includes the necessary parameters for training"""
        return [p for head in self.model.heads for p in head.parameters()]

    def _model_train(self, t):
        """Manage which parameters can be learned and which are frozen"""
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        # freeze feature extractor and only allow the classifier to learn
        self.model.freeze_backbone()
        self.model.model.eval()
        for head in self.model.heads:
            head.train()
            if self.bias:
                head.weight.requires_grad = False

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # Create the joint dataset
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)

        # Load model for task t
        self.model.load_state_dict(torch.load(os.path.join(self.load_dir, 'models', f'task{t}.ckpt'), map_location=self.device))

        # Reinit heads
        if not self.bias:
            for head in self.model.heads:
                head.reset_parameters()

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model_train(t)
        for images, targets in trn_loader:
            # Forward current model
            # --- This first loss learns on unbalanced dataset the current task
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss
