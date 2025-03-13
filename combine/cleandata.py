import os 
import pandas as pd
import numpy as np
import nums_from_string as nfs

path = "E:/張/LAML"
file_name = "combination2.csv"
data=pd.read_csv(os.path.join(path,file_name))

data['years_of_follow_up'] = ''
data['age'] = ''
data['os_time'] = ''
data['agegroup1'] = ''
data['t_depth_group'] = ''
data['n_depth_group'] = ''
for index ,row in data.iterrows():

    if row['race'] == "not reported":
        data= data.drop(index=index)
        continue
    
    if not pd.isnull(row['days_to_last_follow_up']):
        data.at[index,'years_of_follow_up'] = round(row['days_to_last_follow_up']/365.25)
    
    if not pd.isnull(row['age_at_index']):
        data.at[index,'age'] = round(row['age_at_index'])
        if data.at[index,'age'] <= 50:
            data.at[index, 'agegroup1'] = 0
        else:
            data.at[index, 'agegroup1'] = 1

    if row['gender']=="female":
        data.at[index, 'gender'] = 0
    elif row['gender']=="male":
        data.at[index, 'gender'] = 1
    else:
        data.at[index, 'gender'] = ''

    if row['prior_malignancy'] == "yes":
        data.at[index, 'prior_malignancy'] = 1
    elif row['prior_malignancy'] == "no":
        data.at[index, 'prior_malignancy'] = 0
    else:
        data.at[index, 'prior_malignancy'] = ''

    if row['ethnicity'] == "hispanic or latino":
        data.at[index, 'ethnicity'] = 0
    elif row['ethnicity'] == "not hispanic or latino":
        data.at[index, 'ethnicity'] = 1
    else:
        data.at[index, 'ethnicity'] = ''

    if row['prior_treatment'] == "Yes":
        data.at[index, 'prior_treatment'] = 1
    elif row['prior_treatment'] == "No":
        data.at[index, 'prior_treatment'] = 0
    else:
        data.at[index, 'prior_treatment'] = ''

    if row['vital_status'] == "Alive":
        data.at[index, 'os'] = 0
        data.at[index,'os_time'] =data.at[index,'days_to_last_follow_up']
    else:
        data.at[index, 'os'] = 1
        data.at[index,'os_time'] =data.at[index,'days_to_death']

    if row['PICK'] == 1:
        data.at[index, 'PICK'] = 1
    else :
        data.at[index, 'PICK'] = 0

    if row['race'] == "asian":
        data.at[index, 'race'] = 0
    elif row['race'] == "black or african american":
        data.at[index, 'race'] = 1
    elif row['race'] == "white":
        data.at[index, 'race'] = 2
    else:
        data.at[index, 'race'] = ''

    if row['BIOTYPE']=="protein_coding":
        data.at[index, 'BIOTYPE'] = 0
    else:
        data.at[index, 'BIOTYPE'] = 1

    if 'tolerat' in str(row['SIFT']):
        data.at[index, 'SIFT'] = 0
    elif 'deleterious' in str(row['SIFT']):
        data.at[index, 'SIFT'] = 1
    else:
        data.at[index, 'SIFT'] = ''

    if 'benign' in str(row['PolyPhen']):
        data.at[index, 'PolyPhen'] = 0
    elif 'damaging' in str(row['PolyPhen']):
        data.at[index, 'PolyPhen'] = 1
    else:
        data.at[index, 'PolyPhen'] = 2

    if row['IMPACT']=="MODERATE":
        data.at[index, 'IMPACT'] = 0
    elif row['IMPACT']=="LOW":
        data.at[index, 'IMPACT']=1
    else:
        data.at[index, 'IMPACT']=2

    if not pd.isnull(row['t_depth']):
        if data.at[index,'t_depth'] < 183:
            data.at[index, 't_depth_group'] = 0
        else:
            data.at[index, 't_depth_group'] = 1

    if not pd.isnull(row['n_depth']):
        if data.at[index,'n_depth'] < 151:
            data.at[index, 'n_depth_group'] = 0
        else:
            data.at[index, 'n_depth_group'] = 1

select_column = data[['submitter_id', 'dbSNP_RS','age', 'prior_malignancy', 'prior_treatment' , 'gender' , 'race' ,'agegroup1','PICK' ,'SIFT','PolyPhen','t_depth','n_depth','t_depth_group','n_depth_group', 'os', 'os_time']]
select_column.to_csv(os.path.join(path, 'cleaned_file_by_python.csv'), index=False)

"""#平均
select_column['SIFT'] = pd.to_numeric(select_column['SIFT'], errors='coerce')

select_column['PICK'] = select_column.groupby('submitter_id')['PICK'].transform('mean')
select_column['SIFT'] = select_column.groupby('submitter_id')['SIFT'].transform('mean')
select_column['PolyPhen'] = select_column.groupby('submitter_id')['PolyPhen'].transform('mean')
select_column['t_depth'] = select_column.groupby('submitter_id')['t_depth'].transform('mean')
select_column['n_depth'] = select_column.groupby('submitter_id')['n_depth'].transform('mean')
select_column = select_column.groupby('submitter_id').first().reset_index()
select_column.to_csv(os.path.join(path, 'cleaned_file_by_python_avg.csv'), index=False)

select_column = data[['submitter_id' , 'age','dbSNP_RS', 'gender' ,'agegroup1' ,'SIFT','PolyPhen','t_depth','n_depth','PICK', 'prior_malignancy', 'prior_treatment' , 'race', 'os', 'os_time']]
select_column = select_column.dropna()
max_t_depth_index = select_column.groupby('submitter_id')['t_depth'].idxmax()

# 保留 't_depth' 最大值的行
select_column = select_column.loc[max_t_depth_index]
select_column['t_depth'] = select_column.groupby('submitter_id')['t_depth'].transform('max')
select_column['n_depth'] = select_column.groupby('submitter_id')['n_depth'].transform('max')
select_column = select_column.groupby('submitter_id').first().reset_index()
select_column.to_csv(os.path.join(path, 'cleaned_file_by_python_high.csv'), index=False)

# 保留 't_depth' 最小值的行
max_t_depth_index = select_column.groupby('submitter_id')['t_depth'].idxmin()
select_column = select_column.loc[max_t_depth_index]
select_column['t_depth'] = select_column.groupby('submitter_id')['t_depth'].transform('max')
select_column['n_depth'] = select_column.groupby('submitter_id')['n_depth'].transform('max')
select_column = select_column.groupby('submitter_id').first().reset_index()
select_column.to_csv(os.path.join(path, 'cleaned_file_by_python_min.csv'), index=False)"""