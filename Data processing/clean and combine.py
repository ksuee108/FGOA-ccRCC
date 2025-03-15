import pandas as pd 
import os 
import numpy as np
from fancyimpute import IterativeImputer
# Read CSV files
file_mutation = "Mutation_TCGA-KIRC.csv"
file_clinical = "Clinical_TCGA-KIRC.csv"
path = ".\FGOA-ccRcc"

df_mutation = pd.read_csv(os.path.join(path, file_mutation))
df_clinical = pd.read_csv(os.path.join(path, file_clinical))

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
    combin.to_csv('.\\combination2.csv', index=False)  # Added .csv extension
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
        for group_name, value in race_mapping.items():
            if group_name in combin['race'].values:
                group = combin.groupby('race').get_group(group_name)
                clan.loc[group.index, 'race'] = value


        combin['treatments_pharmaceutical_treatment_or_therapy'] = combin['treatments_pharmaceutical_treatment_or_therapy'].replace('not reported', np.nan)
        
        grouped_combin = combin.groupby('treatments_pharmaceutical_treatment_or_therapy').size()

        for x, label in enumerate(grouped_combin.index):
            print("start_gender:", label, x)
            clan.loc[combin['treatments_pharmaceutical_treatment_or_therapy'] == label, "treatments_pharmaceutical_treatment_or_therapy"] = x
            print("end_gender:", x)

        print("treatments_radiation_treatment_or_therapy")
        combin['treatments_radiation_treatment_or_therapy'] = combin['treatments_radiation_treatment_or_therapy'].replace('not reported', np.nan)
        grouped_combin = combin.groupby('treatments_radiation_treatment_or_therapy').size()

        for x, label in enumerate(grouped_combin.index):
            print("start_gender:", label, x)
            clan.loc[combin['treatments_radiation_treatment_or_therapy'] == label, "treatments_radiation_treatment_or_therapy"] = x
            print("end_gender:", x)
        
        ajcc_stage = {
            'Stage I': 0,
            'Stage II': 0,
            'Stage III': 1,
            'Stage IV': 1,
            }
        grouped_combin = combin.groupby('ajcc_pathologic_stage').size()
        for x in range(len(grouped_combin)): 
            print("start_gender:",grouped_combin.index[x],x)
            clan.loc[combin.ajcc_pathologic_stage == grouped_combin.index[x],"ajcc_pathologic_stage"]=x
            print("end_gender:",x)

        for group_name, value in ajcc_stage.items():
            if group_name in combin['ajcc_pathologic_stage'].values:
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

        for group_name, value in ajcc_mapping.items():
            if group_name in combin['ajcc_t'].values:
                group = combin.groupby('ajcc_t').get_group(group_name)
                clan.loc[group.index, 'ajcc_t'] = value

        grouped_data = combin.groupby('ajcc_pathologic_m')
        for group_name, value in ajcc_m_mapping.items():
            if group_name in grouped_data.groups:
                group = grouped_data.get_group(group_name)
                clan.loc[group.index, 'ajcc_pathologic_m'] = value

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

        row['BIOTYPE'] = 1 if 'protein_coding' in str(row['BIOTYPE']) else 0
        row['CLIN_SIG'] = 1 if 'pathogenic' in str(row['CLIN_SIG']) else 0
        row['hotspot'] = 1 if 'Y' in str(row['hotspot']) else 0
        
        return row
    
    def clan(self, df_mutation):
        df_mutation = df_mutation[(df_mutation['CANONICAL']=='YES') & (df_mutation['Variant_Classification']=='Missense_Mutation')]
        df_mutation['Barcode'] = df_mutation['Matched_Norm_Sample_Barcode'].str[:12]
        df_mutation = df_mutation.sort_values('Tumor_Sample_Barcode')
        df_mutation = df_mutation.apply(self.process_row, axis=1)

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

select_column = data[['submitter_id', 'gender', 'treatments_pharmaceutical_treatment_or_therapy', 'treatments_radiation_treatment_or_therapy',
                      'age', 'ajcc_pathologic_stage', 't_depth', 't_alt_count', 'CLIN_SIG', 'hotspot', 'SIFT', 'PolyPhen', 
                      'os', 'os_time']]
select_column = select_column.dropna()

select_column.to_csv(os.path.join(path, 'cleaned_file_stage.csv'), index=False)
select_column.to_csv( ".\\original/stage/cleaned_file_stage.csv", index=False)

select_column = data[['submitter_id', 'gender', 'race', 'treatments_pharmaceutical_treatment_or_therapy', 'treatments_radiation_treatment_or_therapy', 'agegroup',
                      'age', 'ajcc_t', 'ajcc_pathologic_m', 'ajcc_pathologic_n', 't_depth', 't_ref_count', 't_alt_count', 'n_depth', 'BIOTYPE', 'CLIN_SIG', 
                      'hotspot', 'SIFT', 'PolyPhen', 'os', 'os_time']]
select_column = select_column.dropna()

select_column.to_csv(os.path.join(path, 'cleaned_file_TMN.csv'), index=False)
select_column.to_csv( ".\\original/TMN/cleaned_file_TMN.csv", index=False)