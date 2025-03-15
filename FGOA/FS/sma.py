# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:42:56 2024

@author: User1201
"""

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

def init_position(lb, ub, N, dim):
    print(f"N: {N}, dim: {dim}")
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def jfs(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9    # inertia weight
    #vb = 0     # vibration parameter
   # z = 0.1    # random parameter
   # ub       = opts['ub']
    #lb        = opts['lb']
    N = opts['N']
    #print(N)
    max_iter = opts['T']
    #w = opts['w']
    vb = opts['vb']
    z = opts['z']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    same_score_count = 0
    prev_fitG = fitG

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        # Store result
        curve[0, t] = fitG.copy()
        print("Iteration:", t + 1)
        print("Best (SMA):", curve[0, t])
        t += 1

        # Update positions
        for j in range(N):
            if rand() < w:
                # Approach the best solution
                best_pos = Xgb[0]
                X[j] = X[j] + rand() * (best_pos - X[j])
            else:
                # Explore randomly
                random_index = np.random.randint(0, N)
                random_pos = X[random_index]
                X[j] = X[j] + z * rand() * (random_pos - X[j])

            # Vibrate randomly
            X[j] = X[j] + vb * np.random.randn(dim)

            # Ensure positions stay within bounds
            for d in range(dim):
                X[j, d] = boundary(X[j, d], lb[0, d], ub[0, d])
            #判斷是否有重複
            if fitG == prev_fitG:
                same_score_count += 1
                print(same_score_count)
            else:
                same_score_count = 0
                prev_fitG = fitG
        
            if same_score_count >= 10:
                print(f"Generation {t + 1}: Skipping due to repeated scores.")
                # Skip this generation
                t += 1
            break   






    # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    
    # Create dictionary
    sma_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    return sma_data
