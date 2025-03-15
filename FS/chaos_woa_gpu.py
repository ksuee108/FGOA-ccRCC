# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:05:37 2023

@author: user
"""

import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import rand
from FS.functionHO_gpu import Fun

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = 4

def init_position(lb, ub, N, dim):
    X = torch.rand(N, dim, device=device) * (ub - lb) + lb
    return X

def binary_conversion(X, thres):
    Xbin = (X > thres).int()
    return Xbin

def chaotic_map(x, max_val):
    return (a * x * (1 - x)) * max_val

def boundary(x, lb, ub):
    return torch.clamp(x, lb, ub)

def jfs(xtrain, ytrain, opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    thres = 0.5
    b = 1
    ub = torch.tensor(opts['ub'], device=device)
    lb = torch.tensor(opts['lb'], device=device)
    N = opts['N']
    max_iter = opts['T']

    dim = xtrain.shape[1]
    if len(lb) == 1:
        ub = ub.expand(1, dim)
        lb = lb.expand(1, dim)

    X = init_position(lb, ub, N, dim).to(device)
    Xbin = binary_conversion(X, thres)
    fit = torch.zeros(N, 1, device=device)
    Xgb = torch.zeros(1, dim, device=device)
    fitG = float('inf')

    for i in range(N):
        Xbin[i, :] = binary_conversion(X[i, :], thres)
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < fitG:
            Xgb[0, :] = X[i, :]
            fitG = fit[i, 0]

    curve = torch.zeros(1, max_iter, device=device)
    t = 0

    curve[0, t] = fitG.item()
    print("Generation:", t + 1)
    print("Best (WOA):", curve[0, t].item())
    t += 1

    while t < max_iter:
        a = 2 - t * (2 / max_iter)
        a = chaotic_map(a, 1)
        for i in range(N):
            A = 2 * a * torch.rand(1, device=device) - a
            C = 2 * torch.rand(1, device=device)
            p = torch.rand(1, device=device)
            l = -1 + 2 * torch.rand(1, device=device)

            if p < 0.5:
                if torch.abs(A) < 1:
                    for d in range(dim):
                        Dx = torch.abs(C * Xgb[0, d] - X[i, d])
                        X[i, d] = Xgb[0, d] - A * Dx
                        X[i, d] = boundary(X[i, d], lb[0, d].item(), ub[0, d].item())
                elif torch.abs(A) >= 1:
                    for d in range(dim):
                        k = torch.randint(0, N, (1,), device=device)
                        Dx = torch.abs(C * X[k, d] - X[i, d])
                        X[i, d] = X[k, d] - A * Dx
                        X[i, d] = boundary(X[i, d], lb[0, d].item(), ub[0, d].item())
            elif p >= 0.5:
                for d in range(dim):
                    dist = torch.abs(Xgb[0, d] - X[i, d])
                    X[i, d] = dist * torch.exp(b * l) * torch.cos(2 * np.pi * l) + Xgb[0, d]
                    X[i, d] = boundary(X[i, d], lb[0, d].item(), ub[0, d].item())

        Xbin = binary_conversion(X, thres)

        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        curve[0, t] = fitG.item()
        print("Generation:", t + 1)
        print("Best (ChaosWOA):", curve[0, t].item())
        t += 1

    Gbin = binary_conversion(Xgb, thres)
    Gbin = Gbin.view(dim)
    pos = torch.arange(0, dim, device=device)
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    woa_data = {'sf': sel_index.cpu().numpy(), 'c': curve.cpu().numpy(), 'nf': num_feat}
    return woa_data


