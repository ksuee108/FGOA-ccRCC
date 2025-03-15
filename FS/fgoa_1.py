# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 00:35:44 2024

@author: User1201
"""

import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from FS.functionHO import Fun  # 假設這是適應度函數的導入

# Re-defining the Particle class to be used in the AdjustedIPSO algorithm
class Goose:
    def __init__(self, dim, minx, maxx, best_position=None, updrafts=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.updrafts = updrafts

        if best_position is None:
            self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        else:
            self.position = self.initialize_previous_best(dim, best_position)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf
        self.score = np.inf

    def initialize_previous_best(self, dim, best_position):
        return np.copy(best_position)

    def use_updraft(self):
        num_updrafts = np.random.randint(1, 4)
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=2.0)
            distance = np.linalg.norm(self.position - updraft_center)
            if distance < 1.0:
                self.velocity += strength * (updraft_center - self.position) / distance

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
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

    def whiffling_exploitation(self, particle):
        direction_to_gbest = self.gbest - particle.position
        random_factor = np.random.rand() * np.abs(direction_to_gbest)
        new_direction = direction_to_gbest + random_factor
        particle.position = new_direction
        particle.position = np.clip(particle.position, self.minx, self.maxx)

    def update_geese(self, exploitation_prob=0.1, neighborhood_radius=1.5):
        for particle in self.geese:
            if np.random.rand() < exploitation_prob:
                self.whiffling_exploitation(particle)

            neighbors = [p for p in self.geese if np.linalg.norm(p.position - particle.position) < neighborhood_radius]
            if neighbors:
                positions = [p.position for p in neighbors]
                print("Positions of neighbors:", positions)
                if all(pos.shape == positions[0].shape for pos in positions):
                    avg_position = np.mean(positions, axis=0)
                    #avg_position = np.mean([p.position for p in neighbors], axis=0)
                    print("avg_position shape:", avg_position.shape)
                    print("particle.position shape:", particle.position.shape)               
                    particle.velocity += 0.01 * (avg_position - particle.position)
                for neighbor in neighbors:
                    if np.linalg.norm(neighbor.position - particle.position) < neighborhood_radius / 2:
                        particle.velocity -= 0.9 * (neighbor.position - particle.position)

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            particle.velocity = self.inertia * particle.velocity + self.cognitive * r1 * (particle.best_position - particle.position) + self.social * r2 * (self.gbest - particle.position)
            particle.position += particle.velocity + 0.1 * np.random.randn(self.dim)
            particle.position = np.clip(particle.position, self.minx, self.maxx)
            particle.use_updraft()

    def evaluate(self, xtrain, ytrain, opts):
        # Dimension

        X= init_position( self.minx,  self.maxx,  self.size,  self.dim)
        for particle in self.geese:
            Xbin = binary_conversion(X, 0.5, self.size, self.dim)
            particle.score = Fun(xtrain, ytrain, Xbin[0], opts)
            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_position = np.copy(particle.position)

    def whiffling_search(self, particle, xtrain, ytrain, opts):
        new_position = particle.position + np.random.uniform(low=-0.01, high=0.01, size=self.dim)
        new_position = np.clip(new_position, self.minx, self.maxx)
        X= init_position( self.minx,  self.maxx,  self.size,  self.dim)
        Xbin = binary_conversion(X, 0.5, self.size, self.dim)
        new_score = Fun(xtrain, ytrain, Xbin[0], opts)
        if new_score < particle.score:
            particle.position = new_position
            particle.score = new_score
            if new_score < particle.best_score:
                particle.best_score = new_score
                particle.best_position = np.copy(new_position)

    def incentive(self):
        self.incentive_threshold = np.abs(np.mean([p.score for p in self.geese]) - self.gbest_score)
        for particle in self.geese:
            if particle.score < self.incentive_threshold * self.gbest_score:
                r = np.random.rand(self.dim)
                self.gbest = r * self.gbest + (1 - r) * particle.position

    def assist_lagging_geese(self):
        avg_score = np.mean([p.score for p in self.geese])
        threshold = avg_score * 1.9
        for particle in self.geese:
            if particle.score > threshold:
                particle.position += 0.1 * (self.gbest - particle.position)

    def optimize(self, xtrain, ytrain, opts):
        fatigue_counter = 0
        stagnant_counter = 0
        for i in range(self.iter):
            self.evaluate(xtrain, ytrain, opts)
            self.set_leader()
            self.incentive()
            self.update_geese()
            self.best_scores.append(self.gbest_score)

            if i % 100 == 0:
                self.assist_lagging_geese()

            if stagnant_counter >= 100:
                for particle in self.geese:
                    self.whiffling_search(particle, xtrain, ytrain, opts)

            if self.leader.score == self.leader.best_score:
                fatigue_counter += 1
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0

            if fatigue_counter >= self.fatigue:
                self.set_leader()

        Gbin = binary_conversion(np.array([self.gbest]), 0.5, self.size, self.dim).reshape(self.dim)
        pos = np.arange(self.dim)
        sel_index = pos[Gbin == 1]
        num_feat = len(sel_index)
        return {'sf': sel_index, 'c': self.best_scores, 'nf': num_feat}
