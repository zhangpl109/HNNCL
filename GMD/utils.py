import pprint
import random
import sys
import time

import numpy as np
import torch


def cur_hms():
    return time.strftime('%H:%M:%S')


def seed_all(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def training_stats(args, models, train_data, train_data_iter, vocab):
    models = '\n'.join(str(x) for x in models)
    return (f'\n'
            f'argv: {sys.argv[1:]}'
            f'\n\n'
            f'config:\n{pprint.pformat(vars(args), indent=2)}'
            f'\n\n'
            f'train_dataset_size: {len(train_data)}\n'
            f'   train_batch_num: {len(train_data_iter)}\n'
            f'        vocab_size: {len(vocab)}'
            f'\n\n'
            f'Model:\n{models}'
            f'\n\n'
            )


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']
