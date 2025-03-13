import pandas as pd 
import os 
import numpy as np
from fancyimpute import IterativeImputer
# Read CSV files
file_mutation = "Mutation_TCGA-KIRC.csv"
file_clinical = "Clinical_TCGA-KIRC.csv"
path = "D:/張/TCGA-kirc5"

df_mutation = pd.read_csv(os.path.join(path, file_mutation))
df_clinical = pd.read_csv(os.path.join(path, file_clinical))

# Combine clinical and mutation data
# Combine clinical and mutation data
def combine_data(mutation_cleaned, clinical_cleaned):
    # Convert the 'ajcc_pathologic_n' and 'Barcode' columns to string
    clinical_cleaned['bcr_patient_barcode'] = clinical_cleaned['bcr_patient_barcode'].astype(str)
    mutation_cleaned['Barcode'] = mutation_cleaned['Barcode'].astype(str)
    """clinical_cleaned['age_at_index']=df_clinical['age_at_index']
    clinical_cleaned['agegroup'] = clinical_cleaned['agegroup']"""
    combin = pd.merge(clinical_cleaned, mutation_cleaned, left_on='bcr_patient_barcode', right_on='Barcode', how='inner')
    combin = combin.drop(columns=['bcr_patient_barcode','Barcode'])
    combin = combin.dropna()
    combin.to_csv('D:/張/TCGA-kirc5/combination2.csv', index=False)  # Added .csv extension
    return combin

# Continue with the rest of your code as before...


class Clinical:
    def clean(self, combin):
        combin = combin[combin['race'] != "not reported"].copy()
        clan = pd.DataFrame(combin['submitter_id'])
        combin['os_time'] = ''
        combin['agegroup'] = ''
        combin['pharmaceutical_treatment'] = ''
        combin['radiation_treatment'] = ''
        combin['T_stage'] = ''
        combin['M_stage'] = ''
        combin['N_stage'] = ''
        combin['os']=''

        gender = combin.groupby('gender').size()
        for x in range(len(gender)): 
            print("start_gender:",gender.index[x],x)
            clan.loc[combin.gender == gender.index[x],"gender"]=x
            print("end_gender:",x)

        grouped_data = combin.groupby('vital_status')
        group = grouped_data.get_group('Alive')
        clan.loc[group.index, 'os'] = 0
        clan.loc[group.index, 'os_time'] = combin.loc[group.index, 'days_to_last_follow_up']
        group = grouped_data.get_group('Dead')
        clan.loc[group.index, 'os'] = 1
        clan.loc[group.index, 'os_time'] = combin.loc[group.index, 'days_to_death']

        race_mapping = {
                'asian': 1,
                'white': 0,
                'black or african american': 0
            }
        # 遍历映射并进行赋值
        for group_name, value in race_mapping.items():
            if group_name in combin['race'].values:  # 检查该组是否存在于组合数据中
                group = combin.groupby('race').get_group(group_name)
                clan.loc[group.index, 'race'] = value

        """race = combin.groupby('race').size()
        for x in range(len(race)): 
            print("start_gender:",race.index[x],x)
            clan.loc[combin.race == race.index[x],"race"]=x
            print("end_gender:",x)"""
        print("treatments_pharmaceutical_treatment_or_therapy")
        # 现在可以按新的类别进行分组
        combin['treatments_pharmaceutical_treatment_or_therapy'] = combin['treatments_pharmaceutical_treatment_or_therapy'].replace('not reported', np.nan)
        
        """
        # 步骤 1: 替换空值为 'no' 根据另一列是 'yes' 的情况
        clan.loc[(clan['treatments_pharmaceutical_treatment_or_therapy'] == 'yes') & 
                pd.isna(clan['treatments_radiation_treatment_or_therapy']), 
                'treatments_radiation_treatment_or_therapy'] = 'no'

        clan.loc[(clan['treatments_radiation_treatment_or_therapy'] == 'yes') & 
                pd.isna(clan['treatments_pharmaceutical_treatment_or_therapy']), 
                'treatments_pharmaceutical_treatment_or_therapy'] = 'no'
        
        # 步骤 2: 将其余 NaN 替换为 'not reported'
        clan['treatments_pharmaceutical_treatment_or_therapy'] = clan['treatments_pharmaceutical_treatment_or_therapy'].fillna('not reported')
        clan['treatments_radiation_treatment_or_therapy'] = clan['treatments_radiation_treatment_or_therapy'].fillna('not reported')
        """
        grouped_combin = combin.groupby('treatments_pharmaceutical_treatment_or_therapy').size()

        # 遍历 'grouped_combin' 的索引并在 'clan' 中进行映射
        for x, label in enumerate(grouped_combin.index):
            print("start_gender:", label, x)
            clan.loc[combin['treatments_pharmaceutical_treatment_or_therapy'] == label, "treatments_pharmaceutical_treatment_or_therapy"] = x
            print("end_gender:", x)

        # 现在可以按新的类别进行分组
        print("treatments_radiation_treatment_or_therapy")
        combin['treatments_radiation_treatment_or_therapy'] = combin['treatments_radiation_treatment_or_therapy'].replace('not reported', np.nan)
        grouped_combin = combin.groupby('treatments_radiation_treatment_or_therapy').size()
        
        # 遍历 'grouped_combin' 的索引并在 'clan' 中进行映射
        for x, label in enumerate(grouped_combin.index):
            print("start_gender:", label, x)
            clan.loc[combin['treatments_radiation_treatment_or_therapy'] == label, "treatments_radiation_treatment_or_therapy"] = x
            print("end_gender:", x)
        """clan.loc[(clan['treatments_pharmaceutical_treatment_or_therapy'] == 1) & 
                (clan['treatments_radiation_treatment_or_therapy'] == np.nan), 
                'treatments_radiation_treatment_or_therapy'] = 0
        
        clan.loc[(clan['treatments_radiation_treatment_or_therapy'] == 1) & 
                (clan['treatments_pharmaceutical_treatment_or_therapy'] == np.nan), 
                'treatments_pharmaceutical_treatment_or_therapy'] = 0"""
        
        ajcc_stage = {
            'Stage I': 0,
            'Stage II': 0,
            'Stage III': 1,
            'Stage IV': 1,
            }
        grouped_combin = combin.groupby('ajcc_pathologic_stage').size()
        for x in range(len(grouped_combin)): 
            """if grouped_combin.index[x]=="NA":
                #clan.loc[combin.ajcc_pathologic_stage == combin.index[x],"ajcc_pathologic_stage"]=x-1
                continue"""
            print("start_gender:",grouped_combin.index[x],x)
            clan.loc[combin.ajcc_pathologic_stage == grouped_combin.index[x],"ajcc_pathologic_stage"]=x
            print("end_gender:",x)

        for group_name, value in ajcc_stage.items():
            if group_name in combin['ajcc_pathologic_stage'].values:  # 检查该组是否存在于组合数据中
                group = combin.groupby('ajcc_pathologic_stage').get_group(group_name)
                clan.loc[group.index, 'ajcc_pathologic_stage'] = value

        
        ajcc_mapping = {
            'T1': 0,
            'T1a': 0,
            'T1b': 0,
            'T2': 1,
            'T2a': 1,
            'T2b': 1,
            'T3': 2,
            'T3a': 2,
            'T3b': 2,
            'T3c': 2,
            'T4': 3
            }
        
        ajcc_m_mapping = {
            'M0': 0,
            'M1': 1,
            'MX': 0
            }

        ajcc_n_mapping = {
                'N0': 0,
                'N1': 1,
                'NX': 0
            }
        # 遍历映射并进行赋值
        for group_name, value in ajcc_mapping.items():
            if group_name in combin['ajcc_t'].values:  # 检查该组是否存在于组合数据中
                group = combin.groupby('ajcc_t').get_group(group_name)
                clan.loc[group.index, 'ajcc_t'] = value

        # 处理 ajcc_pathologic_m 列
        grouped_data = combin.groupby('ajcc_pathologic_m')
        for group_name, value in ajcc_m_mapping.items():
            if group_name in grouped_data.groups:
                group = grouped_data.get_group(group_name)
                clan.loc[group.index, 'ajcc_pathologic_m'] = value

        # 处理 ajcc_pathologic_n 列
        grouped_data = combin.groupby('ajcc_pathologic_n')
        for group_name, value in ajcc_n_mapping.items():
            if group_name in grouped_data.groups:
                group = grouped_data.get_group(group_name)
                clan.loc[group.index, 'ajcc_pathologic_n'] = value

        clan = clan.sort_values('submitter_id')
        clan['bcr_patient_barcode'] = combin['bcr_patient_barcode']
        return clan
        
    def process_row(self, row):
        if not pd.isnull(row['age']):
            row['agegroup'] = 0 if row['age'] <= 65 else 1
        return row

class Mutation:
    def process_row(self, row):
        # Process SIFT
        """row['SIFT'] = 1 if 'tolerat' in str(row['SIFT']) else 0 
    
        # Process PolyPhen
        row['PolyPhen'] = 1 if 'benign' in str(row['PolyPhen']) else 0 """

        # Process BIOTYPE
        row['BIOTYPE'] = 1 if 'protein_coding' in str(row['BIOTYPE']) else 0
        # Process CLIN_SIG
        row['CLIN_SIG'] = 1 if 'pathogenic' in str(row['CLIN_SIG']) else 0
        # Process hotspot
        row['hotspot'] = 1 if 'Y' in str(row['hotspot']) else 0
        
        return row
    
    def clan(self, df_mutation):
        df_mutation = df_mutation[(df_mutation['CANONICAL']=='YES') & (df_mutation['Variant_Classification']=='Missense_Mutation')]
        df_mutation['Barcode'] = df_mutation['Matched_Norm_Sample_Barcode'].str[:12]
        df_mutation = df_mutation.sort_values('Tumor_Sample_Barcode')
        #df_mutation = df_mutation.dropna(subset=['SIFT','PolyPhen'])
        df_mutation = df_mutation.apply(self.process_row, axis=1)
        # Determine dominant value for SIFT
        """count_0 = (df_mutation['SIFT'] == 0).sum()
        count_1 = (df_mutation['SIFT'] == 1).sum()
        dominant_sift_value = 1 if count_1 >= count_0 else 0

        # Determine dominant value for PolyPhen
        count_0 = (df_mutation['PolyPhen'] == 0).sum()
        count_1 = (df_mutation['PolyPhen'] == 1).sum()
        dominant_polyphen_value = 1 if count_1 >= count_0 else 0"""

        # Convert numeric fields
        for column in ['t_depth', 't_ref_count', 't_alt_count', 'n_depth']:
            df_mutation[column] = pd.to_numeric(df_mutation[column], errors='coerce')

        # Save processed data
        mutation_df = pd.DataFrame({'Barcode': df_mutation['Barcode'].unique()})

        # Average numeric fields
        for column in ['SIFT', 'PolyPhen', 't_depth', 't_ref_count', 't_alt_count', 'n_depth']:
            mutation_df[column] = df_mutation.groupby('Barcode')[column].mean().values

        # Count categorical fields
        for column in ['hotspot','BIOTYPE', 'CLIN_SIG']:
            mutation_df[column] = df_mutation.groupby('Barcode')[column].sum().values
        print(mutation_df['BIOTYPE'])
        """
        mutation_df['SIFT'] = dominant_sift_value
        mutation_df['PolyPhen'] = dominant_polyphen_value"""
        return mutation_df

# Process clinical and mutation data
clinical_instance = Clinical()

mutation_instance = Mutation()

clinical_cleaned = clinical_instance.clean(df_clinical)
print(clinical_instance)
imputer = IterativeImputer(max_iter=10, random_state=0)
clinical_cleaned['treatments_pharmaceutical_treatment_or_therapy'] = imputer.fit_transform(clinical_cleaned[['treatments_pharmaceutical_treatment_or_therapy']])
clinical_cleaned['treatments_radiation_treatment_or_therapy'] = imputer.fit_transform(clinical_cleaned[['treatments_radiation_treatment_or_therapy']])

mutation_cleaned = mutation_instance.clan(df_mutation)
mutation_cleaned.to_csv(os.path.join(path, 'mutation_cleaned_test2.csv'), index=False)
clinical_cleaned.to_csv(os.path.join(path, 'clinical_cleaned_test2.csv'), index=False)

clinical_cleaned['age']=df_clinical['age_at_index']
clinical_cleaned = clinical_cleaned.apply(clinical_instance.process_row, axis=1)
data = combine_data(mutation_cleaned, clinical_cleaned)


print(clinical_cleaned)
print(data.columns)
data.to_csv(os.path.join(path, 'test2.csv'), index=False)




#data = data.apply(clinical_instance.process_row, axis=1)
print(data.head())  # 查看数据前5行
select_column = data[['submitter_id', 'gender', 'treatments_pharmaceutical_treatment_or_therapy', 'treatments_radiation_treatment_or_therapy',
                      'age', 'ajcc_pathologic_stage', 't_depth', 't_alt_count', 'CLIN_SIG', 'hotspot', 'SIFT', 'PolyPhen', 
                      'os', 'os_time']]
select_column = select_column.dropna()

select_column.to_csv(os.path.join(path, 'cleaned_file_stage.csv'), index=False)
select_column.to_csv( "D:/張/TCGA-kirc5/original/stage/cleaned_file_stage.csv", index=False)

select_column = data[['submitter_id', 'gender', 'race', 'treatments_pharmaceutical_treatment_or_therapy', 'treatments_radiation_treatment_or_therapy', 'agegroup',
                      'age', 'ajcc_t', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 't_depth', 't_ref_count', 't_alt_count', 'n_depth', 'BIOTYPE', 'CLIN_SIG', 
                      'hotspot', 'SIFT', 'PolyPhen', 'os', 'os_time']]
select_column = select_column.dropna()

select_column.to_csv(os.path.join(path, 'cleaned_file_TMN.csv'), index=False)
select_column.to_csv( "D:/張/TCGA-kirc5/original/TMN/cleaned_file_TMN.csv", index=False)