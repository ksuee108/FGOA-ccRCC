import pandas as pd
import numpy as np
#import anndata as ad
import math
import seaborn as sns
import matplotlib.pyplot as plt 
#from pydeseq2.dds import DeseqDataSet
#from pydeseq2.ds import DeseqStats
#from sanbomics.tools import id_map
#from sanbomics.plots import volcano
# from pydeseq2.utils import load_example_data
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = 'D:\\張\\TCGA-kirc5\\'
# load data
data = pd.read_csv(os.path.join(path, 'cleaned_file_stage.csv'))
counts_df = data.set_index('submitter_id')
print(counts_df)
counts = counts_df
print(counts)
tumor = pd.DataFrame()
benign_tumor = pd.DataFrame()
counts_index = counts.index
print("counts index", counts_index)

X=counts.drop(columns=['os','os_time'])
y=counts['os']
print("X:\n",X.columns, "y:\n",y.name)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#*******************************Lasso**********************************************************************
# Lasso 回歸分析 (包含交叉驗證選擇最佳 \lambda)
lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 100), cv=10, random_state=42).fit(X_train, y_train)
best_alpha = lasso_cv.alpha_

print(f"最佳懲罰參數 (\lambda): {best_alpha:.4f}")

# 使用最佳 \lambda 進行 Lasso 回歸
lasso = Lasso(alpha=best_alpha).fit(X_train, y_train)
coef = lasso.coef_

# 計算 P 值
# 假設性檢定: t 統計量與 P 值
residuals = y_train - lasso.predict(X_train)
sigma_squared = np.var(residuals)
X_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # 加入截距
inv_XTX = np.linalg.inv(X_with_bias.T @ X_with_bias)
se = np.sqrt(np.diag(inv_XTX) * sigma_squared)  # 標準誤差
t_stats = coef / se[1:]  # 去掉截距部分
p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

# *******************************視覺化 Lasso CV 結果*******************************
# Plot LassoCV 跑出的圖
m_log_alphas = -np.log10(lasso_cv.alphas_)  # 對 \lambda 取對數
plt.figure(figsize=(8, 6))
plt.plot(m_log_alphas, lasso_cv.mse_path_.mean(axis=-1), 'r-', label='Mean CV Error')
plt.fill_between(m_log_alphas,
                 lasso_cv.mse_path_.mean(axis=-1) - lasso_cv.mse_path_.std(axis=-1),
                 lasso_cv.mse_path_.mean(axis=-1) + lasso_cv.mse_path_.std(axis=-1),
                 alpha=0.2, label='Std Dev')
plt.axvline(-np.log10(best_alpha), linestyle='--', color='blue', label='Best λ')
plt.xlabel('-log10(λ)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Cross-Validation')
plt.legend()

# *******************************非零係數視覺化*******************************
non_zero_indices = np.where(coef != 0)[0]
non_zero_features = X.columns[non_zero_indices]
non_zero_coef = coef[non_zero_indices]

plt.figure(figsize=(10, 6))
plt.barh(non_zero_features, non_zero_coef, color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Non-zero Coefficients in Lasso')
plt.tight_layout()
#plt.show()

# *******************************ANOVA檔案儲存*******************************
F_values, anova_p_values = f_classif(X, y)
anova_df = pd.DataFrame({'Feature': X.columns, 'F_value': F_values, 'p_value': anova_p_values})
anova_df['Significant'] = anova_df['p_value'].apply(lambda x: '*' if x < 0.05 else '')
anova_df.to_csv(os.path.join(path, "ANOVA.csv"), index=False)
print("ANOVA結果已儲存至 ANOVA.csv")

# *******************************P值檢查與儲存*******************************
p_values_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef, 'p_value': p_values})
p_values_df['Significant'] = p_values_df['p_value'].apply(lambda x: '*' if x < 0.05 else '')
p_values_df.to_csv(os.path.join(path, "LASSO.csv"), index=False)
print("Lasso結果已儲存至 LASSO.csv")

significant_features = p_values_df[p_values_df['p_value'] < 0.05]
plt.figure(figsize=(10, 6))
plt.barh(significant_features['Feature'], significant_features['Coefficient'], color='lightgreen')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Significant Non-zero Coefficients in Lasso')
plt.show()


# *******************************P值檢查與儲存*******************************
path = 'D:\\張\\TCGA-kirc5\\'
data = pd.read_csv(os.path.join(path, 'FGOA\\stage\\FGOA_all_sel_stage.csv'))
X=counts.drop(columns=['os','os_time'])
y=counts['os']