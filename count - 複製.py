import pandas as pd
from collections import Counter
import os
import ast

optimization = ['PSO', 'GA', 'DE', 'WOA', 'HHO', 'SMA']#, 'FGOA'
file = "classifier_clinical_results_stage.csv"

for opt in optimization:
    D_path2 = f"D:\\張\\TCGA-kirc5\\{opt}\\stage"

    # 读取数据
    data = pd.read_csv(os.path.join(D_path2, file))['Feature names']
    print(data)

    # **將 Series 轉為 List**
    select_list = data.tolist()  # 轉成 list
    print(select_list)  # 確認內容
    
    all_features = []
    
    # 解析每一行的特徵，並加入 all_features
    for item in select_list:
        try:
            feature_list = ast.literal_eval(item)  # 將字串轉回 list
            all_features.extend(feature_list)  # 加入特徵列表
        except:
            print(f"解析失敗：{item}")  # 遇到錯誤時提示
    
    # 使用 Counter 計算特徵出現次數
    feature_counts = Counter(all_features)

    # 排序特徵（按出現次數由大到小）
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # 將排序後的結果轉為 DataFrame
    sorted_features_df = pd.DataFrame(sorted_features, columns=['Feature', 'Count'])

    # 儲存 CSV 檔案
    sorted_features_df.to_csv(os.path.join(D_path2, 'counted_sorted_features_stage.csv'), index=False)

    # 打印當前最佳化演算法的結果
    print(sorted_features_df)
    print(f"Total features: {len(sorted_features)}\n")
