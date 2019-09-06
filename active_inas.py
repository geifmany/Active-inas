import json

import numpy as np
from sklearn.model_selection import train_test_split

import active_utils
from models import cifar10_resnet, cifar100_resnet, svhn_resnet
import math

MODEL_CLASSES = {'cifar10resnet': cifar10_resnet.Cifar10Resnet,
              'cifar100resnet': cifar100_resnet.Cifar100Resnet,
              'svhn_resnet': svhn_resnet.SvhnResnet}
QUERY_FUNCS = {'softmax_response': active_utils.softmax_response,
                           'random': active_utils.random,
                           'coreset': active_utils.coreset,
                           'mc_dropout': active_utils.mc_dropout}


def save_dict(filename, dict):
    with open(filename, 'w') as fp:
        json.dump(dict, fp)


def inas(model_cls, structure, train_idx, pool_idx, T=1, search_space=[12, 5]):
    '''
    This function implements the iNAS algorithm. It is iteratively create a list of structures to be evaluated
    and selects the best among it.

    :param model_cls: A class of the model to be used (the model class trains and evaluate the model.)
    :param structure: the structure to start from (noted as A_0 in the paper)
    :param train_idx: The training set indices at this point in active learning
    :param pool_idx:  The pool set indices at this point in active learning
    :param T: number of iterations to run
    :param search_space: a list [i,j] where i is the maximum number of blocks and j is the maximum number of stacks
    :return:
    '''
    model_search = model_cls(train_idx, pool_idx, depth_mult=1)
    for i in range(T):
        next_structures = [[structure[0]] * structure[1]]
        if structure[0] < search_space[0]:
            next_structures.append([structure[0] + 1] * structure[1])

        if structure[1] < search_space[1]:
            next_structures.append([int((structure[0] * structure[1]) / (structure[1] + 1)) + 1] * (structure[1] + 1))
        best_structure = model_search.search_blocks(next_structures)

        if structure[0] == best_structure[0] and structure[1] == len(best_structure):
            break
        structure = [best_structure[0], len(best_structure)]
    return best_structure, structure


def run_active_learning(model_cls, query_func, model_name, budget=50000, init_size=2000, batch_size=2000,
                        const_structure=None, random_seed=1, T_inas=1):

    '''

    :param model_cls: A class of a model that build train and evaluate a network
    :param query_func: A query function to be used
    :param model_name: A string, a name for the result files to be saved
    :param budget: The active learning budget (int)
    :param init_size: the random seed sike $k$ in the paper
    :param batch_size: The active mini-batch size (b in the paper)
    :param const_structure: A list that represents a structure for running baselines without nas, [2,2,2,2] is Resnet18
    :param random_seed: a random seed to be used to standardize the selection of the initial seed (k points) across exps.
    :param T_inas: The number of iteration to run the iNAS algorithm
    '''

    # sample initial seed
    train_idx, pool_idx, _, _ = train_test_split(np.arange(budget),
                                                 np.arange(budget),
                                                 test_size=(budget - init_size),
                                                 random_state=random_seed)
    best_structure = const_structure
    results = {}
    # the initial structure to be used. Denoted as A_0=A(B,1,1) in the paper.
    structure = [1, 1]

    while 1:
        # step 1: architecture search
        if const_structure is None:
            # perform architecture search if constant architecture is not defined
            best_structure, structure = inas(model_cls, structure, train_idx, pool_idx, T_inas)

        # step 2: train model
        model = model_cls(train_idx, pool_idx, structure=best_structure, depth_mult=1)
        model.train()

        results[model.sample_size] = {'acc': model.accuracy, 'max_acc': model.max_accuracy, 'struture': best_structure,
                                      'size': model.model_size}
        print("accuracy at stage {} is {}".format(model.sample_size, model.max_accuracy))
        save_dict("results/{}.json".format(model_name), results)

        # step 3 query new points
        # stopping criterion
        if train_idx.shape[0] == budget:
            break
        if train_idx.shape[0] >= 10000:
            batch_size = 5000
        if batch_size > pool_idx.shape[0]:
            #fix the size of the last batch if needed
            batch_size = pool_idx.shape[0]

        print("Querying new points")
        new_points = query_func(model, batch_size)

        train_idx = np.concatenate((train_idx, pool_idx[new_points]))
        pool_idx = np.delete(pool_idx, new_points)

        del model  # to free memory during next architecture search


if __name__== "__main__":
    '''
    A demo to run active nais on cifar10 dataset with the softmax response query function
    '''
    model = 'cifar10resnet'  # use one of 'cifar10resnet', cifar100resnet' or 'svhn_resnet'
    query = 'softmax_response' # use one of 'softmax_response', 'mc_dropout' or 'coreset'
    model_name = "{}_{}".format(model, query)
    run_active_learning(MODEL_CLASSES[model], QUERY_FUNCS[query], model_name)


