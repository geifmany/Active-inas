from __future__ import print_function
import numpy as np
import time
'''
This file contains all the implemented query functions
'''


def random(model,points=2000):
    '''
    This function randomly selects indices in range([pool size])
    :param model: the trained model f, used only to get the number of points in the pool
    :param points: number of points to query
    :return: array of indices of the queried points
    '''
    m = model.pool_idx.SHAPE[0]
    all_points = np.arange(m)
    np.random.shuffle(all_points)
    return all_points[:points]


def softmax_response(model, points=5000):
    '''

    :param model: the trained model f, used to calculate the softmax responses
    :param points: number of points to query
    :return: array of indices of the queried points
    '''
    confidence_score = model.softmax_response() #calculate the softmax responses of the pool
    argsort_confidence = np.argsort(confidence_score)
    return argsort_confidence[:points]


def coreset(model, points=5000):
    '''

    :param model: the trained model f, used to calculate the softmax responses
    :param points: number of points to query
    :return: array of indices of the queried points
    '''


    m = model.pool_idx.SHAPE[0]
    all_points = np.arange(m)

    dist_mat, pool_mat = model.coreset_mat() #return two distance matrices, train to pool and pool to pool
    dist_mat_min = np.min(dist_mat,0)
    new_points_dist = dist_mat_min

    selected_points = []
    for i in range(points):
        idx = np.argmax(np.minimum(dist_mat_min[all_points], new_points_dist[all_points]))
        selected_points.append(all_points[idx])
        new_points_dist = np.minimum(new_points_dist, pool_mat[idx])
        np.delete(all_points,(idx),0)
    return np.array(selected_points)

def mc_dropout(model, points=5000):
    '''

    :param model: the trained model f, used to calculate the softmax responses
    :param points: number of points to query
    :return: array of indices of the queried points
    '''

    confidence_score = model.mc_dropout()
    argsort_confidence = np.argsort(confidence_score)
    return argsort_confidence[:points]

