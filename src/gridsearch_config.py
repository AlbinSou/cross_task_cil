class GridSearchConfig():
    def __init__(self):
        self.params = {
            'general': {
                'lr_first': [5e-1, 1e-1, 5e-2],
                'lr': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
                'lr_searches': [3],
                'lr_min': 1e-4,
                'lr_factor': 3,
                'lr_patience': 10,
                'clipping': 10000,
                'momentum': 0.9,
                'wd': 0.0002
            },
            'finetuning': {
            },
            'joint': {
            },
            'bal_ft': {
            },
            'bal_joint': {
            },
        }
        self.current_lr = self.params['general']['lr'][0]
        self.current_tradeoff = 0

    def get_params(self, approach):
        return self.params[approach]
