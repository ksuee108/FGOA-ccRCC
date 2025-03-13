import pandas as pd 
import os 
import numpy as np

# 读取CSV文件
file_mutation = "Mutation_TCGA-KIRC.csv"
path = "D:/張/TCGA-KIRC"
df_mutation = pd.read_csv(os.path.join(path, file_mutation))

# 过滤数据
df_mutation = df_mutation[(df_mutation['CANONICAL']=='YES') & (df_mutation['Variant_Classification']=='Missense_Mutation')]

# 创建Barcode字段，取前12位作为唯一识别码
df_mutation['Barcode'] = df_mutation['Matched_Norm_Sample_Barcode'].str[:12]

# 按照Tumor_Sample_Barcode排序
df_mutation = df_mutation.sort_values('Tumor_Sample_Barcode')

# 处理函数，用来处理行中的特定列
def process_row(row):
    if 'tolerat' in str(row['SIFT']):
        row['SIFT'] = 1
    elif 'deleterious' in str(row['SIFT']):
        row['SIFT'] = 0

    if 'benign' in str(row['PolyPhen']):
        row['PolyPhen'] = 1
    elif 'damaging' in str(row['PolyPhen']):
        row['PolyPhen'] = 0
    else:
        row['PolyPhen'] = 1

    if 'protein_coding' in str(row['BIOTYPE']):
        row['BIOTYPE'] = 1
    else:
        row['BIOTYPE'] = 0

    if 'likely_pathogenic' in str(row['CLIN_SIG']):
        row['CLIN_SIG'] = 1
    else:
        row['CLIN_SIG'] = 0

    if 'Y' in str(row['hotspot']):
        row['hotspot'] = 1
    else:
        row['hotspot'] = 0

    return row

# 应用处理函数到df_mutation
df_mutation = df_mutation.apply(process_row, axis=1)

# 将数值型字段转换为数值类型
column_list = ['t_depth', 't_ref_count', 't_alt_count', 'n_depth']
for column in column_list:
    df_mutation[column] = pd.to_numeric(df_mutation[column], errors='coerce')

# 保存处理过的数据到CSV
df_mutation.to_csv(os.path.join(path, 'mutation_test2.csv'), index=False)

# 创建新的DataFrame存储结果
mutation_df = pd.DataFrame()
mutation_df['Barcode'] = df_mutation['Barcode'].unique()

# 计算数值型字段的平均值
for column in column_list:
    grouped_combin = df_mutation.groupby('Barcode')[column].mean()
    mutation_df[column] = mutation_df['Barcode'].map(grouped_combin)

# 计算分类字段的计数
categorical_list = ['BIOTYPE', 'CLIN_SIG', 'hotspot']
for column in categorical_list:
    print(column)
    grouped_combin = df_mutation.groupby('Barcode')[column].sum()
    print(grouped_combin)
    mutation_df[column] = mutation_df['Barcode'].map(grouped_combin)

# 检查结果
print(mutation_df.head())

# 将结果保存到CSV
mutation_df.to_csv(os.path.join(path, 'mutation_test.csv'), index=False)
