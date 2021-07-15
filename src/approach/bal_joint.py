import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, num_epochs_ft=10, multi_loss=False, reinit_heads=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.nepochs_finetuning = num_epochs_ft
        self.multi_loss = multi_loss
        self.reinit_heads = reinit_heads

        self._balanced_ft_phase = False

        self.trn_datasets = []
        self.val_datasets = []

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--num-epochs-ft', default=10, type=int, required=False,
                            help='Number of epochs for balanced training (default=%(default)s)')
        parser.add_argument('--multi-loss', action='store_true', required=False,
                            help='Applies the multi loss (default=%(default)s)')
        parser.add_argument('--reinit-heads', action='store_true', required=False,
                            help='Re-initializes heads to random before balanced training (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self._train_parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _train_parameters(self):
        """Includes the necessary parameters for training"""
        if self._balanced_ft_phase:
            return [p for head in self.model.heads for p in head.parameters()]
        else:
            return self.model.parameters()

    def _model_train(self, t):
        """Manage which parameters can be learned and which are frozen"""
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        if self._balanced_ft_phase:
            # freeze feature extractor and allow classifier to learn
            self.model.freeze_backbone()
            self.model.model.eval()
            for head in self.model.heads:
                head.train()
        else:
            # when not in the extra balanced finetuning phase, train model as usual with feature extractor
            self.model.train()
            for param in self.model.model.parameters():
                param.requires_grad = True

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
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

        # continue training as usual
        super().train_loop(t, trn_loader, val_loader)

        # Step 2: finetuning on classifier only with FE frozen -- but instead of exemplar memory, all memory
        if t > 0:
            print(' * Training balanced finetuning phase')
            self._balanced_ft_phase = True
            orig_nepochs = self.nepochs
            self.nepochs = self.nepochs_finetuning
            if self.reinit_heads:
                for head in self.model.heads:
                    head.reset_parameters()
            super().train_loop(t, trn_loader, val_loader)
            self.nepochs = orig_nepochs
            self._balanced_ft_phase = False

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model_train(t)
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""

        # If multi-loss, then use L_multi
        if self.multi_loss and not self._balanced_ft_phase and t > 0:
            loss = 0.0
            for m in range(len(targets)):
                task_id = (self.model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
                loss += torch.nn.functional.cross_entropy(outputs[task_id][m].unsqueeze(0),
                                                          targets[m].unsqueeze(0) - self.model.task_offset[task_id])
            loss /= len(targets)
        # Otherwise, use L_all
        else:
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss


class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
