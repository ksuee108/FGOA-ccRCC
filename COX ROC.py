#===================

#python version 3.10
#lifelines version 0.27.8
#pandas version 2.1.3

#===================
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

optimization = ["FGOA"]
for opt in optimization:
    file = f"{opt}_all_sel_stage.csv"
    path = f'.\\FGOA\\stage'
    data = pd.read_csv(os.path.join(path, file))
    """data['Pharmaceutical treatment or therapy'] = data['treatments_pharmaceutical_treatment_or_therapy']
    data['Radiation treatment or therapy'] = data['treatments_radiation_treatment_or_therapy']
    data.drop(columns=['Radiation treatment or therapy', 'Pharmaceutical treatment or therapy'], inplace=True)"""


    X = data.drop(['submitter_id'], axis=1)
    y = data['os']
    X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = train_test_split(X,y, random_state=42, test_size=0.2)
    # 繼續執行 Cox 回歸的擬合
    cph = CoxPHFitter()

    # 只使用性別欄位進行回歸
    cph.fit(X_train_KNN, 'os_time', event_col='os', show_progress=True, robust=True)

    # 打印 Cox 回歸的摘要
    print("========")
    cph.print_summary()
    print("========")
    summary_df = cph.summary
    print("\n\n摘要:",summary_df,"\n\n")
    # 获取模型的系数
    coefficients = cph.summary['coef']

    # 輸出模型系數
    print("模型系数:")
    print(coefficients)

    hazard_ratios = cph.summary['exp(coef)']
    p_value = cph.summary['p']

    # 輸出危險比例
    print("\n\n危險比例:")
    print(hazard_ratios)

    df = pd.DataFrame({
    'Feature': X.drop(['os', 'os_time'], axis=1).columns,
        })
    df = pd.concat([df, summary_df.reset_index(drop=True)], axis=1)

# 保存合并后的 DataFrame
    df.to_csv(os.path.join(path, 'coxPH_Feature.csv'), index=False)

    """
    # 預測部分風險
    partial_hazards = cph.predict_partial_hazard(X_test_KNN)

    # 假設您想要預測存活概率
    survival_probabilities = cph.predict_survival_function(X_test_KNN)
    """
    # Plot hazard ratios
    ax = cph.plot(hazard_ratios=True)  # Plot without `fontsize`
    ax.tick_params(axis='both', which='major', labelsize=20)  # Set font size for tick labels
    ax.set_title("Hazard Ratios with 95% Confidence Intervals", fontsize=20)  # Set title font size
    ax.set_xlabel("Hazard Ratio", fontsize=20)  # Set x-axis font size
    ax.set_ylabel("Covariates", fontsize=20)
    plt.show()

    """c_index = cph.score(X_test_KNN, y_test_KNN['os_time'], event_col='os')
    print("C-index:", c_index)"""