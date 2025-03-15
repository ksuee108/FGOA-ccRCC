import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
import os
import tensorflow as tf
from keras import mixed_precision
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score
from FS.sma import jfs as sma
from FS.woa import jfs as woa
from FS.pso import jfs as pso 
from FS.hho import jfs as hho
from FS.ga import jfs as ga
from FS.de import jfs as de
from FS.fgoa_2 import FGOA 

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
path = 'D:\\張\\TCGA-kirc5\\'
# load data
data = pd.read_csv(os.path.join(path, 'cleaned_file_stage.csv'))#, encoding='big5'
data=data.drop(["os_time","submitter_id"],axis=1)
data=data.dropna()
feat=data.drop(["os"],axis=1)#"submitter_id",
label = data["os"]     # label vector
xtrain, xtest, ytrain, ytest = train_test_split(np.array(feat),np.array( label),random_state=42, test_size=0.2)#, stratify=label,shuffle=False
smote = SMOTE(random_state=42)
xtrain, ytrain = smote.fit_resample(xtrain, ytrain)

result_file = 'classifier_clinical_results_stage.csv'
feature_file = 'selected_clinical_features_stage.csv'

# parameter
k    = 0.001     # k-value in learning rate
N    = 50    # number of particles
T    = 100   # maximum number of iterations
w    = 0.9
lb   = 0
ub   = 2
c1   = 2
c2   = 2
vb = 0
z  = 0.1
b  = 1    # constant

fs_algorithms = {
    
    'FGOA': FGOA,
}
"""'GA': ga,
    'PSO': pso,
    'DE': de,
    'WOA': woa,
    'HHO': hho,
    'SMA': sma,"""
all_curves = {}

def feature_selection_and_classification(model_name, model_pack):
    """
    執行特徵選擇、分類並保存結果。
    """
    for j in range(50):
        print("第",j,"次")
        print(f"\n================== {model_name} ==================")
        folder_path = f'{path}\\convergence' 
        os.makedirs(folder_path, exist_ok=True)
        selected_features_list = []

        # 特徵選擇
        fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
        opts = {'k': k, 'fold': fold, 'N': N, 'T': T, 'b': b, 'lb': lb, 'ub': ub, 'w':w, 'vb':vb, 'z':z}
        if model_name!='FGOA':
            sf, score = model_pack(feat, label, opts)
        else:
            fgoa =  FGOA(dim=np.size(feat, 1), size=50, minx=lb, maxx=ub, iter=100, incentive_threshold=0.8, fatigue=5, inertia=1.3, cognitive=0.6, social=0.8)
            fmdl= fgoa.optimize(feat, label,opts)
            sf   =fmdl['sf']
            score = fmdl['c']
        selected_features = feat.columns[sf].to_list()
        selected_features_list.append(selected_features)

        # 模型訓練
        x_train, y_train = xtrain[:, sf], ytrain
        x_valid, y_valid = xtest[:, sf], ytest
        model = CatBoostClassifier(verbose=0, learning_rate=k, n_estimators=100, depth=4, random_seed=42, task_type="GPU")
        model.fit(x_train, y_train)

        #print(convergence_curves)
        
        # 性能評估
        y_pred = model.predict(x_valid)
        results = {
            'Times': j + 1,
            'Classifier': f'{model_name}',
            'Accuracy': accuracy_score(y_valid, y_pred) * 100,
            'Recall': recall_score(y_valid, y_pred) * 100,
            'Precision': precision_score(y_valid, y_pred) * 100,
            'F1 Score': f1_score(y_valid, y_pred) * 100,
            'Feature Size': len(sf),
            'GA score': score.min(),
            'Feature names': str(selected_features)
        }
        df = pd.DataFrame([results])

        # 将数据保存为 CSV 文件
        df.to_csv(os.path.join(path, f'{model_name}\\stage\\{result_file}'), mode='a', header=not os.path.exists(os.path.join(path, f'{model_name}\\stage\\{result_file}')), index=False)
        flat_list = [item for sublist in selected_features_list for item in sublist]

        df = pd.DataFrame(flat_list)
        df.drop_duplicates().T
        # 将数据保存为 CSV 文件
        df.to_csv(os.path.join(path, f'{model_name}\\stage\\{feature_file}'), index=False)

for model_name, model_path in fs_algorithms.items():
    feature_selection_and_classification(model_name, model_path)