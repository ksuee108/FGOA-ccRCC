import pandas as pd 
import os 
import numpy as np

#read.csv
file_mutation = "LAML_Mutation_new.csv"
file_clinical = "LAML_Clinical_new.csv"
path = "E:/張/LAML"

df_mutation = pd.read_csv(os.path.join(path, file_mutation))
df_clinical = pd.read_csv(os.path.join(path, file_clinical))

df_mutation['Barcode'] = ''
#篩選bcr_patient_barcode並創barcode取bcr_patient_barcode前12個字
df_mutation=df_mutation[(df_mutation['Variant_Type']=='SNP')&(df_mutation['Variant_Classification']=='Missense_Mutation')]#&(df_mutation['BIOTYPE']=='protein_coding')

df_mutation['Barcode'] = df_mutation['Matched_Norm_Sample_Barcode'].apply(lambda x:str(x[:12]))
df_mutation = df_mutation.sort_values('Tumor_Sample_Barcode')
df_clinical = df_clinical.sort_values('submitter_id')

combin = pd.merge(df_clinical, df_mutation, left_on='bcr_patient_barcode', right_on='Barcode', how='inner')
data_no_duplicates = combin.drop_duplicates()

values_to_remove = ['No', 'Not Reported', ' ', np.nan,'not reported','Bone marrow']#,'Acute myeloid leukemia, NOS','Unknown','protein_coding','released','Pharmaceutical Therapy, NOS',
                    #'TCGA-LAML','Radiation Therapy, NOS','GRCh38','Somatic','Transcript','missense_variant','MODERATE','+',''

# 遍历每一列，检查是否包含要删除的值，然后删除整列
for column in data_no_duplicates.columns:
    if any(data_no_duplicates[column].isin(values_to_remove)):
        data_no_duplicates = data_no_duplicates.drop(columns=[column])

combin.to_csv(os.path.join(path, 'combination.csv'), index=False)