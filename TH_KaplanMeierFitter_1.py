import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times 
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
import os 

path = r"C:\Users\user\OneDrive - 國立高雄科技大學\文件\GitHub\FGOA-ccRCC"
df = pd.read_csv(os.path.join(path, 'FGOA_all_sel_stage.csv'))
df = df.dropna()
df.drop_duplicates(inplace=True) 

all_df = df[['Pharmaceutical treatment or therapy', 
                            'Radiation treatment or therapy', 'ajcc_pathologic_stage', 'os_time', 'os']] 
with_out_os_time = df[['Pharmaceutical treatment or therapy', 
                            'Radiation treatment or therapy', 'ajcc_pathologic_stage']]

kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))


for col in with_out_os_time.columns:
    print(col)
    group_1 = all_df[all_df[col] == 0]  # no treatment group
    group_2 = all_df[pd.to_numeric(all_df[col], errors='coerce').notna()]  # not reported group
    group_3 = all_df[all_df[col] == 1]  # yes treatment group
    stage1 = all_df[all_df['ajcc_pathologic_stage'] == 0]
    stage2 = all_df[all_df['ajcc_pathologic_stage'] == 1]
    if all_df[col].dtype in ['int64','float64']:
        # Kaplan-Meier survival analysis
        if col !='ajcc_pathologic_stage':
            kmf.fit(durations=group_1['os_time'], event_observed=group_1['os'], label=f"No")
            ax = kmf.plot_survival_function(ax=plt.subplot(111), ci_show=True, color='darkkhaki')
            
            kmf.fit(durations=group_2['os_time'], event_observed=group_2['os'], label=f"not reported")
            kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

            kmf.fit(durations=group_3['os_time'], event_observed=group_3['os'], label=f"Yes")
            kmf.plot_survival_function(ax=ax, ci_show=True)
            
            results = logrank_test(durations_A=group_1["os_time"],
                                durations_B=group_2["os_time"],
                                event_observed_A=group_1["os"],
                                event_observed_B=group_2["os"],
                                )
            results.print_summary()

        else:
            kmf.fit(durations=stage1['os_time'], event_observed=stage1['os'], label=f"I~II")
            ax = kmf.plot_survival_function(ax=plt.subplot(111), ci_show=True, color='darkkhaki')
            
            kmf.fit(durations=stage2['os_time'], event_observed=stage2['os'], label=f"III~IV")
            kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

            results = logrank_test(durations_A=stage1["os_time"],
                                durations_B=stage2["os_time"],
                                event_observed_A=stage1["os"],
                                event_observed_B=stage2["os"],
                                )
            results.print_summary()

        p_value_text = f"P-value = {results.p_value:.3e}"
        ax.text(0.7, 0.1, p_value_text, fontsize=18, fontweight='bold', family='Times New Roman', transform=ax.transAxes)

        plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 18})

        ax.set_xlabel('Time (days)', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Survival Probability', fontsize=18, fontweight='bold', family='Times New Roman')

        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')

        add_at_risk_counts(kmf, ax=ax)
        plt.gcf().axes[-1].tick_params(axis='x', labelsize=18)
        plt.gcf().axes[-1].tick_params(axis='y', labelsize=18)
        for label in plt.gcf().axes[-1].get_xticklabels() + plt.gcf().axes[-1].get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
        plt.show()