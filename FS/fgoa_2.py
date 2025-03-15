# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:14:40 2022

@author: user
"""

import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from FS.functionHO import Fun  # 假設這是適應度函數的導入

class Goose:
    def __init__(self, dim, minx, maxx, best_position=None, updrafts=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.updrafts = updrafts

        if best_position is None:
            self.position = np.random.uniform(low=minx, high=maxx, size=(dim,))
        else:
            self.position = self.initialize_previous_best(dim, best_position)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=(dim,))
        self.best_position = np.copy(self.position)
        self.best_score = np.inf
        self.score = np.inf

    def initialize_previous_best(self, dim, best_position):
        return np.copy(best_position)

    def use_updraft(self):
        num_updrafts = np.random.randint(1, 10)
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=3.0)
            distance = np.linalg.norm(self.position - updraft_center)
            if distance < 1.0:
                self.velocity += strength * ((updraft_center - self.position).reshape(self.velocity.shape)) / distance


def binary_conversion(X, thres=0.5):
    """
    将连续值转换为二进制值。
    
    参数：
    - X: 输入数组，可以是一维或二维。
    - thres: 阈值，默认为0.5。
    
    返回：
    - Xbin: 二进制数组，与X形状相同。
    """
    # 如果X是一维数组，重塑为二维数组
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # 应用阈值转换
    Xbin = (X > thres).astype(int)
    
    # 如果原始X是一维数组，返回一维数组
    if Xbin.shape[0] == 1:
        return Xbin.flatten()
    return Xbin

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

class FGOA:
    def __init__(self, dim, size, minx, maxx, iter, incentive_threshold, fatigue, inertia, cognitive, social):
        self.dim = dim
        self.size = size
        self.minx = minx * np.ones([1, dim], dtype='float')
        self.maxx = maxx * np.ones([1, dim], dtype='float')
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.iter = iter
        self.incentive_threshold = incentive_threshold
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)
        self.gbest_score = np.inf
        self.fatigue = fatigue
        self.best_scores = []

    def set_leader(self):
        scores = [p.best_score for p in self.geese]
        leader_index = scores.index(min(scores))
        self.leader = self.geese[leader_index]
        self.gbest_score = self.leader.best_score
        self.gbest = np.copy(self.leader.best_position)

    def update_geese(self, exploitation_prob=0.1, neighborhood_radius=1.5):
        for particle in self.geese:

            neighbors = [p for p in self.geese if np.linalg.norm(p.position - particle.position) < neighborhood_radius]

            if neighbors:
                positions = [p.position.flatten() for p in neighbors]

                if all(pos.shape == positions[0].shape for pos in positions):
                    avg_position = np.mean(positions, axis=0)
                    avg_position = np.squeeze(avg_position)
                    particle.position = np.squeeze(particle.position)
                    particle.velocity = particle.velocity + 0.01 * (avg_position - particle.position)

                    for neighbor in neighbors:
                        if np.linalg.norm(neighbor.position - particle.position) < neighborhood_radius / 2:
                            particle.velocity -= 0.9 * (neighbor.position - particle.position).squeeze()
                else:
                    print("Error: Neighbor positions have inconsistent shapes")
                    continue

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            particle.velocity = self.inertia * particle.velocity + self.cognitive * r1 * (particle.best_position - particle.position) + self.social * r2 * (self.gbest - particle.position)
            particle.position = particle.position + particle.velocity + 0.1 * np.random.randn(self.dim)
            particle.position = np.clip(particle.position, self.minx, self.maxx)
            particle.use_updraft()

    def evaluate(self, xtrain, ytrain, opts):
        X = init_position(self.minx, self.maxx, self.size, self.dim)
        thres = 0.5
        for particle in self.geese:
            #print("evaluate")
            Xbin = binary_conversion(X, thres)
            particle.score = Fun(xtrain, ytrain, Xbin[0], opts)
            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_position = np.copy(particle.position)

    def incentive(self):
        #self.incentive_threshold = np.abs(np.mean([p.score for p in self.geese]) - self.gbest_score)
        for particle in self.geese:
            if particle.score < self.incentive_threshold * self.gbest_score:
                r = np.random.rand(self.dim)
                self.gbest = r * self.gbest + (1 - r) * particle.position

    def assist_lagging_geese(self):
        avg_score = np.mean([p.score for p in self.geese])
        threshold = avg_score / 1.9
        for particle in self.geese:
            if particle.score < threshold:
                particle.position += 0.1 * (self.gbest - particle.position)

    def optimize(self, xtrain, ytrain, opts):
        
        fatigue_counter = 0
        stagnant_counter = 0
        for i in range(self.iter):
            print("iter:", i)
            self.evaluate(xtrain, ytrain, opts)
            self.set_leader()
            self.incentive()
            self.update_geese()
            self.best_scores.append(self.gbest_score)

            if i % 100 == 0:
                self.assist_lagging_geese()

            if self.leader.score == self.leader.best_score:
                fatigue_counter += 1
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0

                
        thres = 0.5    
        #print("thres",thres,type(thres))
        #print("gbest",self.gbest,self.gbest.shape,type(self.gbest))
        #print("Gbin")
        Gbin = binary_conversion(self.gbest, thres=thres)  # 修改此处
        
        # 由于 binary_conversion 已处理维度，直接使用即可
        sel_index = np.where(Gbin == 1)[0]
        num_feat = len(sel_index)
        return {'sf': sel_index, 'c': np.array(self.best_scores), 'nf': num_feat}
