import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the approach from Workshop CL CVPR 2021"""

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

        if len(self.exemplars_dataset) and t > 0:
            # add exemplars to usual train_loader
            trn_loader = DataLoader(trn_loader.dataset + self.exemplars_dataset, batch_size=trn_loader.batch_size,
                                    shuffle=True, num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # Step 2: finetuning on classifier only with balanced old and new data
        if len(self.exemplars_dataset) and t > 0:
            print(' * Training balanced finetuning phase')
            self._balanced_ft_phase = True
            orig_nepochs = self.nepochs
            self.nepochs = self.nepochs_finetuning
            trn_loader = DataLoader(self.exemplars_dataset, batch_size=trn_loader.batch_size,
                                    shuffle=True, num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
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
