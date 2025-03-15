# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:20:54 2023

@author: user
"""

# Importing required libraries
import numpy as np
from numpy.random import rand
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
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

    # def use_updraft(self):
    #     if self.updrafts:
    #         for updraft_center, strength in self.updrafts:
    #             distance = np.linalg.norm(self.position - updraft_center)
    #             if distance < 2.0:
    #                 self.velocity += strength * (updraft_center - self.position) / distance
                    
    def use_updraft(self):
        # 隨機產生上升氣流的數量
        num_updrafts = np.random.randint(1, 10)  # 假設每次有 1 到 3 個上升氣流

        # 為每個上升氣流隨機產生中心位置和強度
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=2.0)  # 強度範圍可根據需要調整

            # 計算粒子與上升氣流中心的距離
            distance = np.linalg.norm(self.position - updraft_center)

            # 如果粒子在上升氣流的有效範圍內，則根據氣流強度調整速度
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
# Adjusted Goose Theory for High-dimensional Problems
class FGOA:
    def __init__(self, dim, size, minx, maxx, iter, incentive_threshold, fatigue,inertia,cognitive,social):
        self.dim = dim
        self.size = size
        self.minx = minx
        self.maxx = maxx
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.iter = iter
        self.incentive_threshold = incentive_threshold # 激勵閾值
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)
        self.gbest_score = np.inf
        self.fatigue = fatigue
        self.best_scores = []
        # Binary conversion
        #self.Xbin = binary_conversion(X, thres, N, dim)
    def set_leader(self):
        scores = [p.best_score for p in self.geese]
        leader_index = scores.index(min(scores))
        self.leader = self.geese[leader_index]
        self.gbest_score = self.leader.best_score
        self.gbest = np.copy(self.leader.best_position)
        
                
    def whiffling_exploitation(self, particle):      
        # Calculate the direction towards the global best
        direction_to_gbest = self.gbest - particle.position
        
        # Generate a random factor for direction and distance
        random_factor = np.random.rand() * np.abs(direction_to_gbest)
        
        # Add randomness to simulate the "whiffling" behavior
        new_direction = direction_to_gbest + random_factor
        
        # Update the position based on the new direction
        particle.position = new_direction
        
        # Clip to ensure the position stays within bounds
        particle.position = np.clip(particle.position, self.minx, self.maxx)
        
      
    def update_geese(self, exploitation_prob=0.1, neighborhood_radius=1.5):#
        for particle in self.geese:
            # Whiffling Exploitation
            if np.random.rand() < exploitation_prob:
                self.whiffling_exploitation(particle)
    
            # Neighbors Information
            neighbors = [p for p in self.geese if np.linalg.norm(p.position - particle.position) < neighborhood_radius]
            if neighbors:
                # Combine Alignment and Cohesion
                avg_position = np.mean([p.position for p in neighbors], axis=0)
                particle.velocity += 0.01 * (avg_position - particle.position)
    
                # Separation
                for neighbor in neighbors:
                    if np.linalg.norm(neighbor.position - particle.position) < neighborhood_radius / 2:
                        particle.velocity -= 0.9 * (neighbor.position - particle.position)
                        
            
            # Standard Velocity Update
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            #inertia = 0.1#*0.3
            #cognitive = 0.5#*1.5
            #social = 0.5#*1.5
            particle.velocity = self.inertia * particle.velocity + self.cognitive * r1 * (particle.best_position - particle.position) + self.social * r2 * (self.gbest - particle.position)
            particle.position += particle.velocity + 0.1 * np.random.randn(self.dim)  # Slightly increased randomness
            particle.position = np.clip(particle.position, self.minx, self.maxx)
    
            # Use Updraft
            particle.use_updraft()
    
            
    def evaluate(self, func): #要偏移就變成 (self, func, o)
        for particle in self.geese:
            particle.score = func(particle.position)
            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_position = np.copy(particle.position)
                
    def whiffling_search(self, particle, func):
        new_position = particle.position + np.random.uniform(low=-0.01, high=0.01, size=self.dim)
        new_position = np.clip(new_position, self.minx, self.maxx)
        new_score = func(new_position)
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
        # Calculate average score of the flock
        avg_score = np.mean([p.score for p in self.geese])

        # Define a threshold to identify underperforming geese
        threshold = avg_score * 1.9  # Example: 20% worse than average

        # Loop through geese and assist the ones lagging behind
        for particle in self.geese:
            if particle.score > threshold:
                # Adjust position towards the global best or average position of better performers
                # This is a simple example, you might want to use a more sophisticated method
                particle.position += 0.1 * (self.gbest - particle.position)
                
    def visualize(self, func):
        if self.dim > 2:
            print("Visualization is only available for 2D.")
            return

        # Create a mesh grid for contour plot
        x = np.linspace(self.minx, self.maxx, 100)
        y = np.linspace(self.minx, self.maxx, 100)
        X, Y = np.meshgrid(x, y)
        # Z = func(np.array([X, Y]))
        Z = np.array([[func(np.array([x_i, y_i])) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, 20)
        plt.scatter([p.position[0] for p in self.geese], [p.position[1] for p in self.geese], color='red')

        plt.title('Fly Geese in the search space')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def visualize_3d(self, func):
        if self.dim != 3:
            print("Visualization is only available for 3D.")
            return
    
        # Create a 3D mesh grid for contour plot
        x = np.linspace(self.minx, self.maxx, 100)
        y = np.linspace(self.minx, self.maxx, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        
        Z = Z.reshape(X.shape)
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
        ax.scatter([p.position[0] for p in self.geese], 
                   [p.position[1] for p in self.geese], 
                   [func(p.position) for p in self.geese], 
                   color='r', s=100, depthshade=True)
    
        ax.set_title('Fly Geese in the search space')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        plt.show()
        
    def visualize_high_dim(self, positions, title="geese in the 2D visualized space"):
        """
        Visualize high-dimensional particle positions using t-SNE.
        """
        tsne = TSNE(n_components=2)
        reduced_data = tsne.fit_transform(positions)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='red')
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        
    def optimize(self, func):
        fatigue_counter = 0
        stagnant_counter = 0  # Initialize stagnation counter for WLS
        # Binary conversion
        #Xbin = binary_conversion(X, thres, N, dim)
        for i in range(self.iter):
            self.evaluate(func) #要偏移就變成 (func, o)
            self.set_leader()
            self.incentive()
            self.update_geese()
            self.best_scores.append(self.gbest_score)
            
            # Assist lagging geese at certain intervals, e.g., every 100 iterations
            if i % 100 == 0:
                self.assist_lagging_geese()
            
            if i % 100 == 0:  # Update the visualization every 100 iterations
                self.visualize(func)
                # self.visualize_high_dim([p.position for p in self.geese], title=f"Iteration {i}")
            
            # Whiffling Local Search
            if stagnant_counter >= 100:  # If no improvement in last N iterations
                for particle in self.geese:
                    self.whiffling_search(particle, func)
            
            # Fatigue Check and Reset
            if self.leader.score == self.leader.best_score:
                fatigue_counter += 1
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0 # Reset the stagnation counter if any improvement
            
            if fatigue_counter >= self.fatigue:
                self.set_leader()
                
        return self.gbest, self.gbest_score



