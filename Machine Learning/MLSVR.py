import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import os
from Criteria import Evaluation_Criteria
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def SVM_ROC(self):
        """
        Train the model using Support Vector Machine and return the prediction results.

        :return: Prediction results (1D array)
        """
        neigh = SVC(probability=True)
        neigh.fit(self.X_train, self.y_train)
        y_pred = neigh.predict(self.X_test)
        y_pro = neigh.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 

def create_dataset(dataset, look_back, scaler_type='StandardScaler'):
    dataX, dataY = [], []
    scaler = StandardScaler() if scaler_type == 'StandardScaler' else MinMaxScaler()
    features = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    
    features_scaled = scaler.fit_transform(features)
    
    dataset_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    dataset_scaled['label'] = labels.values
    
    for i in range(len(dataset_scaled) - look_back):
        dataX.append(dataset_scaled.iloc[i:(i + look_back), :-1].values)
        dataY.append(dataset_scaled.iloc[i + look_back, -1])
    return np.array(dataX), np.array(dataY)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
optimization = ['original', "GA", "DE", "PSO", "WOA", "HHO", "SMA", "FGOA"]#
all_df = [] 

fig1, bx = plt.subplots()
fig2, dx = plt.subplots()
plt.rcParams['font.family'] = 'Times New Roman'
for opt in optimization:
    print(opt)
    path = f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage'
    file = f'{opt}_all_sel_stage.csv'
    data = pd.read_csv(os.path.join(path, file))
    #data = data.sort_values(by='os_time')
    
    label = data.drop(['os_time', 'submitter_id', 'agegroup'], axis=1) if 'agegroup' in data.columns else data.drop(['os_time', 'submitter_id'], axis=1)
    feature_names = label.drop(['os'],axis=1).columns

    scaler = StandardScaler()
    features = label.iloc[:, :-1]
    labels = label.iloc[:, -1]
    
    features_scaled = scaler.fit_transform(features)

    dataset_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    dataset_scaled['label'] = labels.values

    # Apply SMOTE
    FULL_X_res, FULL_Y_res = SMOTE(random_state=42).fit_resample(features, labels)

    # Split the resampled dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(FULL_X_res, FULL_Y_res, random_state=42, test_size=0.2)
    
    tree_depth, tree_num, learning_rate = 10, 50, 0.001
    # CNN
    print('X_test:',X_test.shape)
    test_y_predicted, predictions= SVM_ROC()

#=============================================================================================
    criteria = Evaluation_Criteria(y_train, y_test, bx, dx, path)

    confusion_result = criteria.confusion(test_y_predicted, predictions, f"{opt} SVC")
    
    # Convert confusion_result to DataFrame
    result_df = pd.DataFrame(confusion_result)  # Wrap confusion_result in a list

    # Append the DataFrame to all_df
    all_df.append(result_df)
    bx.set_xlabel('Recall', fontsize=20, family = 'Times New Roman')
    bx.set_ylabel('Precision', fontsize=20, family = 'Times New Roman')
    bx.set_title('Precision-Recall Curve', fontsize=25, family = 'Times New Roman')
    dx.set_title('ROC Curve', fontsize=25, family = 'Times New Roman')
    dx.set_xlabel('False Positive Rate', fontsize=20, family = 'Times New Roman')
    dx.set_ylabel('True Positive Rate', fontsize=20, family = 'Times New Roman')

    bx.plot([0, 1], [1, 0], lw=2, c='k', linestyle='--')
    dx.plot([0, 1], [0, 1], lw=2, c='k', linestyle='--')

    bx.legend(fontsize = 15)
    dx.legend(fontsize = 15)
    
    # Save each optimization result as a CSV file
    result_df.to_csv(os.path.join(path, f'{opt}_table5 SVC stage.csv'), index=False)
#=============================================================================================

# Concatenate all results
final_df = pd.concat(all_df, axis=0, ignore_index=True)
final_df.to_csv(os.path.join(path, f'table5 SVC stage.csv'), index=False)

criteria.Heatmap(final_df, [f'original SVC', f"GA SVC", f"PSO SVC", f"DE SVC", f"WOA SVC", 
                        f"HHO SVC", f"SMA SVC", f"FGOA SVC"])
print(final_df)
plt.show()
