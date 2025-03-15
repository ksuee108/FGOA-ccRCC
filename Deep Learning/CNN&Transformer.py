import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt 
from keras.layers import LSTM, Dense, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras_self_attention import SeqSelfAttention
import time
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import os
from Criteria import Evaluation_Criteria
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from keras import layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)]  # 可根據需要調整
            )
        print(" GPU 已設定為內存動態增長")
    except RuntimeError as e:
        print(e)


def create_transformer(X_train, X_test, y_train, y_test, opt, input_shape=(10, 15), num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=input_shape)

    # 將維度轉換為 Transformer 的輸入格式
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = layers.Dropout(0.2)(x)
    res = x + inputs

    # 前饋神經網路層 (Feed Forward Layer)
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(input_shape[-1])(x)
    x = layers.Dropout(0.3)(x)
    x = x + res

    # 平坦化輸出，接全連接層
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid", dtype="float32")(x)  # 二分類問題

    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2)

    # 預測
    predictions = model.predict(X_test)
    test_y_predicted = (predictions > 0.5).astype(int)  # 轉換為 0/1
    model.save(f'.\\{opt}\\stage\\{opt} transformer.h5')
    return test_y_predicted, predictions

def cnn(X_train, X_test, y_train, y_test, opt):
    # CNN input shape must be 3D: (samples, timesteps, features)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 二元分類

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    start = time.time()
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0, validation_split=0.2)
    end = time.time()
    
    predictions = model.predict(X_test)
    test_y_predicted = (predictions >= 0.5).astype(int)  # 閾值 0.5

    model.save(f'.\\{opt}\\stage\\{opt} CNN.h5')
    return test_y_predicted, predictions

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def bert(X_train, X_test, y_train, y_test, opt):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 將文本編碼成 BERT 可接受的格式
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

    # 將標籤轉換為 tensor
    train_labels = torch.tensor(y_train.tolist())
    test_labels = torch.tensor(y_test.tolist())
    
    # 創建 PyTorch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    train_dataset = Dataset(train_encodings, train_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    # 載入 BERT 模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir=f'.\\{opt}\\stage\\{opt}-bert',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 訓練模型
    trainer.train()

    # 預測
    predictions = trainer.predict(test_dataset).predictions
    test_y_predicted = predictions.argmax(-1)

    # 保存模型
    model.save_pretrained(f'.\\{opt}\\stage\\{opt}-BERT')
    tokenizer.save_pretrained(f'.\\{opt}\\stage\\{opt}-BERT')
    
    return test_y_predicted, predictions

def create_dataset(dataset, look_back, scaler_type='StandardScaler'):
    dataX, dataY = [], []
    # 選擇標準化方法
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # 只標準化特徵列，跳過標籤列
    features = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    
    # 標準化特徵數據
    features_scaled = scaler.fit_transform(features)
    
    # 將標準化數據轉換回 DataFrame，與標籤合併
    dataset_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    dataset_scaled['label'] = labels.values
    
    for i in range(len(dataset_scaled) - look_back):
        # 選擇範圍內的特徵數據並轉換成數組
        a = dataset_scaled.iloc[i:(i + look_back), :-1].values
        dataX.append(a)
        dataY.append(dataset_scaled.iloc[i + look_back, -1])
    
    return np.array(dataX), np.array(dataY)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
optimization = ['original', "GA", "DE", "PSO", "WOA", "HHO", "SMA", "FGOA"]
lstmmodel = [(create_transformer, 'Transformer')]#(cnn,'CNN'), (bert,'A-BERT')]#

while(1):
    for model, model_name in lstmmodel:
        all_df = [] 
        fig1, bx = plt.subplots()
        fig2, dx = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        for opt in optimization:
            print(opt)
            path = f'.\\{opt}\\stage'
            file = f'{opt}_all_sel_stage.csv'
            data = pd.read_csv(os.path.join(path, file))
            #data = data.sort_values(by='os_time')
            look_back = 10
            
            if 'agegroup' in data.columns:
                label = data.drop(['os_time', 'submitter_id', 'agegroup'], axis=1)  # 修正拼写
            else:
                label = data.drop(['os_time', 'submitter_id'], axis=1)  # 修正拼写

            FULL_X, FULL_Y = create_dataset(label, look_back)
            FULL_X_reshaped = FULL_X.reshape(FULL_X.shape[0], -1)
            smote = SMOTE(random_state=42)
            FULL_X_res, FULL_Y_res = smote.fit_resample(FULL_X_reshaped, FULL_Y)
            FULL_X_res = FULL_X_res.reshape(-1, look_back, FULL_X.shape[2])
            X_train, X_test, y_train, y_test = train_test_split(FULL_X_res, FULL_Y_res, random_state=42, test_size=0.2)

            test_y_predicted, predictions = model(X_train, X_test, y_train, y_test, opt, input_shape = (X_train[1].shape))

            """
            # LSTM 模型训练
            LSTM_binary, LSTM_predictions = model(X_train, X_test, y_train, y_test, opt)
            
            if model_name == 'LSTM':
                model = load_model(f'D:\\張\\TCGA-KIRC\\model\\{opt} LSTM.h5')
            else:
                model = load_model(f'D:\\張\\TCGA-KIRC\\model\\{opt} A-LSTM.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
            LSTM_predictions = model.predict(X_test)
            
            LSTM_binary = (LSTM_predictions >= 0.5).astype(int)
            """
            
            # 绘制图形
            criteria = Evaluation_Criteria(y_train if model_name != 'BERT' else label_train, 
                                        y_test if model_name != 'BERT' else label_test, 
                                        bx, dx, path)
            # 获取混淆结果并存储到 DataFrame
            confusion_result = criteria.confusion(test_y_predicted, predictions, f"{opt} {model_name}")
            
            # Convert confusion_result to DataFrame
            result_df = pd.DataFrame(confusion_result)  # Wrap confusion_result in a list
            #print("result_df", result_df.shape)
            
            # Append the DataFrame to all_df
            all_df.append(result_df)
            #plt.rcParams['font.family'] = 'Times New Roman'
            # Plot settings
            bx.set_xlabel('Recall', fontsize=20, family = 'Times New Roman')
            bx.set_ylabel('Precision', fontsize=20, family = 'Times New Roman')
            bx.set_title('Precision-Recall Curve', fontsize=25, family = 'Times New Roman')
            dx.set_title('ROC Curve', fontsize=25, family = 'Times New Roman')
            dx.set_xlabel('False Positive Rate', fontsize=20, family = 'Times New Roman')
            dx.set_ylabel('True Positive Rate', fontsize=20, family = 'Times New Roman')

            bx.plot([0, 1], [1, 0], lw=2, c='k', linestyle='--')  # 绘制参考线
            dx.plot([0, 1], [0, 1], lw=2, c='k', linestyle='--')

            bx.legend(fontsize = 15)
            dx.legend(fontsize = 15)
            
            # Save each optimization result as a CSV file
            result_df.to_csv(os.path.join(path, f'{opt}_table5 {model_name} stage.csv'), index=False)

        # Concatenate all results
        final_df = pd.concat(all_df, axis=0, ignore_index=True)
        final_df.to_csv(os.path.join(path, f'table5 {model_name} stage.csv'), index=False)

        # 绘制热图
        criteria.Heatmap(final_df, [f'{model_name}', f"{model_name}", f"PSO {model_name}", f"DE {model_name}", f"WOA {model_name}", 
                                    f"HHO {model_name}", f"SMA {model_name}", f"FGOA {model_name}"])
        print(final_df)
        plt.show()
    fgoa_accuracy = final_df[final_df['Model'] == 'FGOA Transformer']['Accuracy'].values[0]

    # 比較並檢查是否大於其他的 Accuracy 值
    for index, row in final_df.iterrows():
        if row['Accuracy'] > fgoa_accuracy:
            print(f"Found model with higher accuracy than FGOA Transformer: {row['Model']} with accuracy {row['Accuracy']}")
            break  # 一旦找到更大的 Accuracy，退出循環