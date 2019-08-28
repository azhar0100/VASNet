__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

from torch.autograd import Variable


class HParameters:

    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15
        self.n_heads = 4
        self.crude_early_stopping = False
        self.crude_early_stopping_n = 60
        self.learning_rate_scheduling = False
        self.use_extra_linear = False

        self.l2_req = 0.00001
        # self.l2_req = 4.192296532832617e-09
        self.lr_epochs = [0]
        self.lr = [0.001]

        self.epochs_max = 300
        self.train_batch_size = 1
        self.train = True

        self.output_dir = 'ex-complete'

        self.root = ''
        self.datasets=['datasets/eccv16_dataset_summe_google_pool5.h5',
                       'datasets/eccv16_dataset_tvsum_google_pool5.h5',
                       'datasets/eccv16_dataset_ovp_google_pool5.h5',
                       'datasets/eccv16_dataset_youtube_google_pool5.h5']

        self.splits = ['splits/tvsum_splits.json',
                        'splits/summe_splits.json']
        # self.splits = ['splits/summe_splits.json']

        # self.splits += ['splits/summe_aug_splits.json']
        self.splits += ['splits/tvsum_aug_splits.json',
                        'splits/summe_aug_splits.json']

        return


    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)

    def literal_load_from_args(self,args):
        for key in args:
            val = args[key]
            if val is not None:
                setattr(self, key, val)

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


if __name__ == "__main__":

    # Tests
    hps = HParameters()
    print(hps)

    args = {'root': 'root_dir',
            'datasets': 'set1,set2,set3',
            'splits': 'split1, split2',
            'new_param_float': 1.23456
            }

    hps.load_from_args(args)
    print(hps)
