"""
Script for running black-box attacks and report their metrics

Note this script makes use of both pytorch and tensor flow to make use of the GPUs
pytorch: for performing the perturbations
tensorflow: for querying the loss/gradient oracle

A good practice is to let pytorch be allocated on GPU:0
and let tensorflow be allocated on GPU:1
CPU mode works fine too.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import time
import sys

import numpy as np
import pandas as pd
# import tensorflow as tf
import torch as ch
import torch.nn as nn


from QueryAttacks.signhunter_original.datasets.my_dataset import Dataset
from QueryAttacks.signhunter_original.utils.compute_fcts import ch_nsign, sign
from QueryAttacks.signhunter_original.utils.helper_fcts import config_path_join, data_path_join, \
    construct_model, get_model_file, create_dir, get_dataset_shape
from QueryAttacks.signhunter_original.utils.latex_fcts import res_json_2_tbl_latex
from QueryAttacks.signhunter_original.utils.plt_fcts import plt_from_h5tbl

from attacks.nes_attack import NESAttack
# from attacks.blackbox.cheat_attack import CheatAttack
from attacks.bandit_attack import BanditAttack
# from attacks.blackbox.zo_sign_sgd_attack import ZOSignSGDAttack
from attacks.sign_attack import SignAttack
# from attacks.blackbox.random_attack import RandAttack
# from attacks.blackbox.naive_attack import NaiveAttack
# from attacks.blackbox.shc_attack import SHCAttack

from QueryAttacks.signhunter_original.victim import load_net, load_net_gs
import sys
import torch
import os

# to run the attacks on a quadratic function with no constraint
# i.e. a concave fct with a single global solution

# class Logger(object):
#     def __init__(self, logFile="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(logFile, 'a')

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         pass

def test_clean_acc(model, dset, config):
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    print('Iterating over {} batches'.format(num_batches))
    acc = 0
    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

        x_batch, y_batch = dset.get_eval_data(bstart, bend)
        bacc = model.accuracy(x_batch, y_batch)
        acc += (bacc*(bend - bstart))
    acc /= 10000
    print('clean accuray:%.4f'%(acc))
    return acc

IS_DEBUG_MODE = False


class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''

    def __init__(self, params):
        self.params = params

    def __getattr__(self, x):
        return self.params[x.lower()]


def attack_mode(model, gpu, batsi=100, args=None, cfg=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    #sys.stdout = Logger(logpath)
    exp_id = 'sign_cifar'
    print("Running Experiment {} with DEBUG MODE {}".format(exp_id, IS_DEBUG_MODE))
    if cfg == None:
        cfs = ['cifar10-nes-linf-config.json', 'cifar10-sign-linf-config.json']  #'cifar10_bandit_linf_config_a.json','cifar10-nes-linf-config.json'
    else:
        cfs = [cfg]

    # create/ allocate the result json for tabulation
    data_dir = data_path_join('blackbox_attack_exp')
    create_dir(data_dir)
    res = {}

    # create a store for logging / if the store is there remove it
    store_name = os.path.join(data_dir, '{}_tbl.h5'.format(exp_id))
    offset = 0
    # rewrite all the results alternatively one could make use of `offset` to append to the h5 file above.
    if os.path.exists(store_name):
        os.remove(store_name)

    for _cf in cfs:
        # for reproducibility
        np.random.seed(1)
        config_file = config_path_join(_cf)
        print(f"config path: {config_file}")

        with open(config_file) as config_file:
            # config = json.load(config_file)
            defaults = json.load(config_file)
            config = defaults
            # arg_vars = vars(args)
            # arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
            # defaults.update(arg_vars)
            # config = defaults
            # args = Parameters(config)
            # args_dict = config

        print(f"config: {config}")
        #### cifar-10 #######
        dset = Dataset(config['dset_name'], config['dset_config'])
        dset_dim = np.prod(get_dataset_shape(config['dset_name']))
        #####################
        device = ch.device(config['device'])
        model = load_net_gs(model, device, args)
        ### imagenet ####
        if args.dataset == 'imagenet':
            def load_imagenet(n_ex, model, seed=0):
                if args.model_type == 'vanilla':
                    data_path = 'data/imagenet_wide_resnet50_2_imgs_0.npy'
                    label_path = 'data/imagenet_wide_resnet50_2_lbls_0.npy'
                elif args.model_type == 'AT':
                    data_path = 'data/imagenet_wide_resnet50_2_AT_imgs_0.npy'
                    label_path = 'data/imagenet_wide_resnet50_2_AT_lbls_0.npy'
                elif args.model_type == 'RND':
                    data_path = 'data/imagenet_wide_resnet50_2_RND_imgs_0.npy'
                    label_path = 'data/imagenet_wide_resnet50_2_RND_lbls_0.npy'
                elif args.model_type == 'UniG':
                    data_path = 'data/imagenet_wide_resnet50_2_imgs_0.npy'
                    label_path = 'data/imagenet_wide_resnet50_2_lbls_0.npy'
                elif args.model_type == 'UniGAT':
                    data_path = 'data/imagenet_wide_resnet50_2_AT_imgs_0.npy'
                    label_path = 'data/imagenet_wide_resnet50_2_AT_lbls_0.npy'
                if not os.path.exists(data_path) or not os.path.exists(label_path):
                    print('Data Path False')
                else:
                    x_test = np.load(data_path)
                    y_test = np.load(label_path)
                return x_test[:n_ex], y_test[:n_ex]
            x_test, y_test = load_imagenet(config["num_eval_examples"], model)
        #################

        # set torch default device:
        if 'cuda' in config['device'] and ch.cuda.is_available():
            ch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            ch.set_default_tensor_type('torch.FloatTensor')

        # saver = tf.train.Saver()
        attacker = eval(config['attack_name'])(
            **config['attack_config'],
            lb=dset.min_value,
            ub=dset.max_value
        )

        # to over-ride attacker's configuration
        attacker.max_loss_queries = args.max_query
        attacker.eval_batch_size = args.batch_size
        if args.dataset == 'cifar10':
            attacker.epsilon = args.test_eps*255
        elif args.dataset == 'imagenet':
            attacker.epsilon = args.test_eps

        print('batch_size:{:d}, max_query:{:d}, eps:{:.3f}'
              .format(attacker.eval_batch_size, attacker.max_loss_queries, attacker.epsilon))

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        # eval_batch_size = config['eval_batch_size']
        eval_batch_size = batsi
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        print('Iterating over {} batches'.format(num_batches))
        start_time = time.time()
        adv_acc = 0
        acc = 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

            if args.dataset == 'cifar10':
                x_batch, y_batch = dset.get_eval_data(bstart, bend)
            elif args.dataset == 'imagenet':
                x_batch, y_batch = x_test[bstart:bend, :], y_test[bstart:bend].astype(np.float32())
            else:
                print('Batch Data False')
            print('clean acc: ', model.accuracy(x_batch, y_batch))

            def loss_fct(xs):
                _l = model.y_xent(xs, y_batch)
                return _l

            def early_stop_crit_fct(xs):
                _is_correct = model.correct_prediction(xs, y_batch)
                # print(np.mean(_is_correct))
                # sys.stdout.flush()
                return np.logical_not(_is_correct)

            logs_dict, adv = attacker.run(x_batch, loss_fct, early_stop_crit_fct)
            # adv = adv.detach().cpu().numpy()
            # print('attack norm: ', np.linalg.norm((adv-x_batch)[0].reshape(-1), ord = np.inf))
            # bacc = model.accuracy(adv, y_batch)
            # acc = acc + (bacc*(bend - bstart))
            # print('adv acc: ', bacc)

            _len = len(logs_dict['iteration'])
            if _len != 0:  # to save i/o ops
                logs_dict['p'] = [config['attack_config']['p']] * _len
                logs_dict['attack'] = [config['attack_name']] * _len
                logs_dict['dataset'] = [config['dset_name']] * _len
                logs_dict['batch_id'] = ibatch
                logs_dict['idx'] = [_ + offset for _ in range(_len)]
                offset += _len
                pd.DataFrame(logs_dict).set_index('idx').to_hdf(store_name, 'tbl', append=True,
                                                                min_itemsize={'p': 3, 'attack': 30, 'dataset': 10})
            print(attacker.summary())

        print("Batches done after {} s".format(time.time() - start_time))
        # acc = acc / num_eval_examples
        # print('adv classification accuray:%.4f'%(acc))

        if config['dset_name'] not in res:
            res[config['dset_name']] = [attacker.summary()]
        else:
            res[config['dset_name']].append(attacker.summary())

    # create latex table
    res_fname = os.path.join(data_dir, '{}_res.json'.format(exp_id))
    print("Storing tabular data in {}".format(res_fname))
    with open(res_fname, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)

    res_json_2_tbl_latex(res_fname)
    # plt_from_h5tbl([store_name])

    return

