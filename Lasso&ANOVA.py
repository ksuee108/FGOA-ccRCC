import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
path = 'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\'
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#*******************************Lasso**********************************************************************
# Lasso regression with cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 100), cv=10, random_state=42).fit(X_train, y_train)
best_alpha = lasso_cv.alpha_

print(f"best lambda : {best_alpha:.4f}")

lasso = Lasso(alpha=best_alpha).fit(X_train, y_train)
coef = lasso.coef_

# *******************************Calculate standard error, t-values, and p-values*******************************
residuals = y_train - lasso.predict(X_train)
sigma_squared = np.var(residuals)
X_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
inv_XTX = np.linalg.inv(X_with_bias.T @ X_with_bias)
se = np.sqrt(np.diag(inv_XTX) * sigma_squared)
t_stats = coef / se[1:]
p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

# *******************************Visualize Lasso CV results*******************************
# Plot LassoCV results
m_log_alphas = -np.log10(lasso_cv.alphas_)
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

# *******************************Visualize non-zero coefficients*******************************
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

# *******************************Save ANOVA results*******************************
F_values, anova_p_values = f_classif(X, y)
anova_df = pd.DataFrame({'Feature': X.columns, 'F_value': F_values, 'p_value': anova_p_values})
anova_df['Significant'] = anova_df['p_value'].apply(lambda x: '*' if x < 0.05 else '')
anova_df.to_csv(os.path.join(path, "ANOVA.csv"), index=False)
print("ANOVA results saved to ANOVA.csv")

# *******************************Check and save p-values*******************************
p_values_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef, 'p_value': p_values})
p_values_df['Significant'] = p_values_df['p_value'].apply(lambda x: '*' if x < 0.05 else '')
p_values_df.to_csv(os.path.join(path, "LASSO.csv"), index=False)
print("Lasso results saved to LASSO.csv")

significant_features = p_values_df[p_values_df['p_value'] < 0.05]
plt.figure(figsize=(10, 6))
plt.barh(significant_features['Feature'], significant_features['Coefficient'], color='lightgreen')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Significant Non-zero Coefficients in Lasso')
plt.show()
