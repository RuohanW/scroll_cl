# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'seq-cifar100': {
        'scroll' :{
            200: {'lr':0.003, 'lr_cl':0.5, 'er_iters': 350, 'alpha':0.001, 'scale_init': 1,
                  'batch_size': 128, 'minibatch_size': 50, 'n_epochs': 1},
            500: {'lr':0.003, 'lr_cl':0.5, 'er_iters': 1200, 'alpha':0.001, 'scale_init': 1,
                  'batch_size': 128, 'minibatch_size': 50, 'n_epochs': 1},
            2000: {'lr':0.01, 'lr_cl':0.5, 'er_iters': 2500, 'alpha':0.01, 'scale_init': 5,
                  'batch_size': 128, 'minibatch_size': 50, 'n_epochs': 1}
        }
    }
}