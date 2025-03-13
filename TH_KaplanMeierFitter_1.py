import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times 
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
import os 

path = 'D:\\張\\TCGA-KIRC5\\FGOA\\stage'
df = pd.read_csv(os.path.join(path, 'FGOA_all_sel_stage.csv'))
df = df.dropna()  # 删除缺失值
df.drop_duplicates(inplace=True)  # 删除重复项

# 提取相关列
all_df = df[['Pharmaceutical treatment or therapy', 
                            'Radiation treatment or therapy', 'ajcc_pathologic_stage', 'os_time', 'os']] 
with_out_os_time = df[['Pharmaceutical treatment or therapy', 
                            'Radiation treatment or therapy', 'ajcc_pathologic_stage']]

# 对每个治疗类型进行生存分析
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

# 遍历每一列，进行生存分析
for col in with_out_os_time.columns:
    print(col)
    group_1 = all_df[all_df[col] == 0]  # 没有治疗的组
    group_2 = all_df[pd.to_numeric(all_df[col], errors='coerce').notna()]  # not reported的组
    group_3 = all_df[all_df[col] == 1]  # 有治疗的组
    stage1 = all_df[all_df['ajcc_pathologic_stage'] == 0]
    stage2 = all_df[all_df['ajcc_pathologic_stage'] == 1]
    if all_df[col].dtype in ['int64','float64']:
        # 对每一列进行分组
        group_1 = all_df[all_df[col] == 0]  # 没有治疗的组
        group_2 = all_df[pd.to_numeric(all_df[col], errors='coerce').notna()]  # not reported的组
        group_3 = all_df[all_df[col] == 1]  # 有治疗的组
        age_1 = all_df[all_df['ajcc_pathologic_stage'] ==0]
        age_2 = all_df[all_df['ajcc_pathologic_stage'] ==1]
        # Kaplan-Meier生存分析
        if col !='ajcc_pathologic_stage':
            kmf.fit(durations=group_1['os_time'], event_observed=group_1['os'], label=f"No")
            ax = kmf.plot_survival_function(ax=plt.subplot(111), ci_show=True, color='darkkhaki')
            
            kmf.fit(durations=group_2['os_time'], event_observed=group_2['os'], label=f"not reported")
            kmf.plot_survival_function(ax=ax, ci_show=True, color='red')
            
            print("group_3 数据类型：")
            print(group_3.dtypes)
            print("\ngroup_3 缺失值统计：")
            print(group_3.isnull().sum())
            print("\ngroup_3 数据示例：")
            print(group_3.head())

            kmf.fit(durations=group_3['os_time'], event_observed=group_3['os'], label=f"Yes")
            kmf.plot_survival_function(ax=ax, ci_show=True)
            
            results = logrank_test(durations_A=group_1["os_time"],
                                durations_B=group_2["os_time"],
                                event_observed_A=group_1["os"],
                                event_observed_B=group_2["os"],
                                )
            results.print_summary()

        else:
            kmf.fit(durations=age_1['os_time'], event_observed=age_1['os'], label=f"I~II")
            ax = kmf.plot_survival_function(ax=plt.subplot(111), ci_show=True, color='darkkhaki')
            
            kmf.fit(durations=age_2['os_time'], event_observed=age_2['os'], label=f"III~IV")
            kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

            results = logrank_test(durations_A=age_1["os_time"],
                                durations_B=age_2["os_time"],
                                event_observed_A=age_1["os"],
                                event_observed_B=age_2["os"],
                                )
            results.print_summary()

        p_value_text = f"P-value = {results.p_value:.3e}"  # 将p值格式化为科学计数法
        ax.text(0.7, 0.1, p_value_text, fontsize=18, fontweight='bold', family='Times New Roman', transform=ax.transAxes)

        # 放大圖例標籤，使用 Times New Roman 並加粗字體
        plt.legend(prop={'size': 15, 'weight': 'bold', 'family': 'Times New Roman'})

        # 設定 x 和 y 軸標籤字體為 Times New Roman 並加粗字體，字體大小為 18
        ax.set_xlabel('Time (days)', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Survival Probability', fontsize=18, fontweight='bold', family='Times New Roman')

        # 設定 x 和 y 軸刻度字體為 Times New Roman 並加粗字體，字體大小為 18
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')

        # 可選地，在圖下方添加 at risk 計數，並設置字體
        add_at_risk_counts(kmf, ax=ax)
        #plt.gcf().axes[-1].set_xlabel('At risk', fontsize=18, fontweight='bold', family='Times New Roman')
        plt.gcf().axes[-1].tick_params(axis='x', labelsize=18)
        plt.gcf().axes[-1].tick_params(axis='y', labelsize=18)
        for label in plt.gcf().axes[-1].get_xticklabels() + plt.gcf().axes[-1].get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
            
        #plt.savefig("./image/TH_treatments_radiation_treatment_or_therapy_KaplanMeierFitter_3.png", format='png', dpi=600)

        plt.show()