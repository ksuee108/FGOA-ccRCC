# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:14:40 2022

@author: user
"""

import numpy as np

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
#from FS.fgoa_3 import jfs   # change this to switch algorithm 
from FS.fgoa_2 import FGOA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
import sklearn
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from imblearn.over_sampling import SMOTE

mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已设置 GPU 内存动态增长。")
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = 'D:\\張\\TCGA-CHOL'

# load data
data = pd.read_csv(os.path.join(path, 'miRNAcom.csv'))#, encoding='big5'
data=data.drop(["submitter_id"],axis=1)
data=data.dropna()

x = data.drop(["os"], axis=1)
y = data["os"]
xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), np.array(y), random_state=42, test_size=0.2)

# Define optimization methods and their configurations
methods = [FGOA]
for method in methods:
    method_name = method.__name__  # Get method name as a string
    result_file = f'{method_name}_classifier_clinical_results.csv'
    selected_features_file = f'{method_name}_selected_clinical_features.csv'
    selected_features_list = []

    for j in range(50):
        print("第",j,"次")
        fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}#.astype(int)

        # parameter
        k    = 0.001     # k-value in KNN
        N    = 50    # number of particles
        T    = 100   # maximum number of iterations
        w    = 0.9
        lb   = 0
        ub   = 2
        c1   = 2
        c2   = 2
        b    = 1     # constant

        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b,'lb':lb,'ub':ub}

    # perform feature selection
        fgoa =  FGOA(dim=np.size(x, 1), size=50, minx=lb, maxx=ub, iter=100, incentive_threshold=0.8, fatigue=5, inertia=1.3, cognitive=0.6, social=0.8)
        fmdl= fgoa.optimize(x, y,opts)#
        #adjusted_gbest_ackley, adjusted_gbest_score_ackley = fgoa.optimize(aic)

        #fmdl = jfs(feat, label, opts)
        #sf   = adjusted_gbest_score_ackley
        sf   =fmdl['sf']
        score = fmdl['c']
        min_score = score.min()
        selected_features = x.columns[sf]
        selected_features_list.append(selected_features.to_list())

        
        # model with selected features
        num_train = np.size(xtrain, 0)
        num_valid = np.size(xtest, 0)
        x_train   = xtrain[:, sf]
        y_train   = ytrain.reshape(num_train)  # Solve bug
        x_valid   = xtest[:, sf]
        y_valid   = ytest.reshape(num_valid)  # Solve bug

        mdl = CatBoostClassifier(verbose = 0,learning_rate=0.01,n_estimators=100,depth=10,random_seed=10,task_type="GPU")#,task_type="GPU"

        mdl.fit(x_train, y_train)

        #mdl.fit(x_train.astype(int), y_train.astype(int))#
    # accuracy
        y_pred    = mdl.predict(x_valid)
        accuracy_score = sklearn.metrics.accuracy_score(y_valid, y_pred) * 100
        recall_score = sklearn.metrics.recall_score(y_valid, y_pred) * 100
        precision_score = sklearn.metrics.precision_score(y_valid, y_pred) * 100
        F1_score = sklearn.metrics.f1_score(y_valid, y_pred) * 100

        print("Accuracy:",accuracy_score)
        print("recall:", recall_score)
        print("precision:", precision_score)
        print("F1_score:", F1_score)
        num_feat = fmdl['sf']
    # plot convergence
        """curve   = fmdl['c']
        curve   = curve.reshape(np.size(curve,1))
        x       = np.arange(0, opts['T'], 1.0) + 1.0

        fig, ax = plt.subplots()
        ax.plot(x, curve, 'o-')
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title('GA')
        ax.grid()
        
        # 保存图表而不是显示
        plt.savefig(f'convergence_plot_{j}.png')
        plt.close()"""
        data = {
            'Times': j,
            'Classifier': ['KNN'],
            'Accuracy': [accuracy_score],
            'Recall': [recall_score],
            'Precision': [precision_score],
            'F1 Score': [F1_score],
            'Feature Size': [len(num_feat)],
            'GA score': [min_score],  # 将单个值包装为列表
            'Feature names': [selected_features_list[-1]]  # 只保存當前的特徵集
        }
        # 创建一个 DataFrame
        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by='Accuracy', ascending=False)
        # 将数据保存为 CSV 文件
        df_sorted.to_csv(os.path.join(path, result_file), mode='a', header=not os.path.exists(os.path.join(path, result_file)), index=False)
        flat_list = [item for sublist in selected_features_list for item in sublist]

        df = pd.DataFrame(flat_list)
        df.drop_duplicates().T
        # 将数据保存为 CSV 文件
        df.to_csv(os.path.join(path, 'selected_clinical_features_stage.csv'), index=False)

    print("特徵選取存檔完畢")