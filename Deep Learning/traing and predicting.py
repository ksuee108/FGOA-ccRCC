import pandas as pd 
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt 
from keras.layers import LSTM, Dense, Dropout, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras_self_attention import SeqSelfAttention
import time
from keras.models import Sequential
import os
from Criteria import Evaluation_Criteria
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import shap
from tensorflow import keras 
from keras import layers
from keras.callbacks import ModelCheckpoint

mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def a_lstm(X_train, X_test, y_train, y_test, opt):
    # LSTM input shape must be 3D: (samples, timesteps, features)
    model = Sequential()
    
    # Both LSTM layers return sequences to maintain 3D input for attention
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(1024, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.2))#, kernel_regularizer=l2(0.002)
    model.add(LSTM(32, activation='sigmoid', return_sequences=True))
    
    # Attention layer
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    
    # Global Average Pooling to reduce dimensions
    model.add(GlobalAveragePooling1D())
    
    # Final Dense layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = ModelCheckpoint(
            filepath=f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage\\{opt} A-LSTM.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=2
        )

    model.fit(X_train, y_train, epochs=1000, batch_size=128 , callbacks=callbacks, verbose=2, validation_split=0.2)
    predictions = model.predict(X_test)
    test_y_predicted = (predictions >= 0.5).astype(int)  # Binary classification threshold at 0.5
    criteria = Evaluation_Criteria(y_train, y_test, bx, dx, path)

    confusion_result = criteria.confusion(test_y_predicted ,predictions, f"{opt} {model_name}")
    result_df = pd.DataFrame(confusion_result)

    model2 = load_model(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage\\{opt} A-LSTM.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
    predictions2 = model2.predict(X_test)
    test_y_predicted2 = (predictions2 >= 0.5).astype(int)  # Binary classification threshold at 0.5
    criteria = Evaluation_Criteria(y_train, y_test, bx, dx, path)

    confusion_result2 = criteria.confusion(test_y_predicted2 ,predictions2, f"{opt} {model_name}")
    result_df2 = pd.DataFrame(confusion_result2)
    
    if result_df.iloc[0]['Accuracy'] > result_df2.iloc[0]['Accuracy']:
        model.save(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage\\{opt} A-LSTM.h5')
        return test_y_predicted, predictions
    else:
        return test_y_predicted2, predictions2

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
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.2)
    end = time.time()
    
    predictions = model.predict(X_test)
    test_y_predicted = (predictions >= 0.5).astype(int)  # 閾值 0.5

    model.save(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage\\{opt} CNN.h5')
    return test_y_predicted, predictions

def create_transformer(X_train, X_test, y_train, y_test, opt, input_shape=(10, 15), num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=input_shape)
    inputs = tf.cast(inputs, tf.float16)

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = layers.Dropout(0.2)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(input_shape[-1])(x)
    x = layers.Dropout(0.3)(x)
    x = x + res

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=2, validation_split=0.2)

    # predict
    predictions = model.predict(X_test)
    test_y_predicted = (predictions > 0.5).astype(int)
    model.save(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage\\{opt} transformer.h5')
    return test_y_predicted, predictions

def create_dataset(dataset, look_back, scaler_type='StandardScaler'):
    dataX, dataY = [], []
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    features = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    
    features_scaled = scaler.fit_transform(features)
    

    dataset_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    dataset_scaled['label'] = labels.values
    
    for i in range(len(dataset_scaled) - look_back):
        dataX.append(dataset_scaled.iloc[i:(i + look_back), :-1].values)
        dataY.append(dataset_scaled.iloc[i + look_back, -1])
    
    return np.array(dataX), np.array(dataY)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
optimization = ['original', 'LASSO', 'ANOVA', "GA", "DE", "PSO", "WOA", "HHO", "SMA", "FGOA"]# 
lstmmodel = [(a_lstm,'A-LSTM'), (cnn, 'CNN'), (create_transformer, 'Transformer')]#

while(1):
    for model, model_name in lstmmodel:
        all_df = []
        fig1, bx = plt.subplots()
        fig2, dx = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        for opt in optimization:

            print(f'======================================={opt}=======================================')
            path = f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\{opt}\\stage'
            file = f'{opt}_all_sel_stage.csv'
            data = pd.read_csv(os.path.join(path, file))
            look_back = 10
            data = data.dropna()
            label = data.drop(['os_time', 'submitter_id', 'agegroup'], axis=1) if 'agegroup' in data.columns else data.drop(['os_time', 'submitter_id'], axis=1)
            feature_names = label.drop(['os'],axis=1).columns
            FULL_X, FULL_Y = create_dataset(label, look_back)
            FULL_X_reshaped = FULL_X.reshape(FULL_X.shape[0], -1)

            # Apply SMOTE
            FULL_X_res, FULL_Y_res = SMOTE(random_state=42).fit_resample(FULL_X_reshaped, FULL_Y)

            # Reshape FULL_X back to 3D for LSTM input
            FULL_X_res = FULL_X_res.reshape(-1, look_back, FULL_X.shape[2])

            # Split the resampled dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(FULL_X_res, FULL_Y_res, random_state=42, test_size=0.2)
            X_train_2d = X_test[:, -10, :]
            X_test_2d = X_test[:, -10, :]
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            #=======================train or call back model=======================
            """
            # train model
            if model_name != 'Transformer':
                model_binary, model_prediction = model(X_train, X_test, y_train, y_test, opt)
            else:    
                model_binary, model_prediction = model(X_train, X_test, y_train, y_test, opt, input_shape = (X_train[1].shape))
            
            """
            # call back model
            if model_name != 'A-LSTM':
                model = load_model(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\model\\{opt} {model_name}.h5')
            else:
                model = load_model(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\model\\{opt} {model_name}.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
            model_prediction = model.predict(X_test)
            
            model_binary = (model_prediction >= 0.5).astype(int)

            #=========================Plot heatmap, ROC, and PRC===================================
            criteria = Evaluation_Criteria(y_train, y_test, bx, dx, path)

            confusion_result = criteria.confusion(model_binary ,model_prediction, f"{opt} {model_name}")
            
            result_df = pd.DataFrame(confusion_result)
            
            # Append the DataFrame to all_df
            all_df.append(result_df)
            # Plot settings
            bx.set_xlabel('Recall', fontsize=20, family = 'Times New Roman')
            bx.set_ylabel('Precision', fontsize=20, family = 'Times New Roman')
            bx.set_title('Precision-Recall Curve', fontsize=25, family = 'Times New Roman')
            dx.set_title('ROC Curve', fontsize=25, family = 'Times New Roman')
            dx.set_xlabel('False Positive Rate', fontsize=20, family = 'Times New Roman')
            dx.set_ylabel('True Positive Rate', fontsize=20, family = 'Times New Roman')

            bx.plot([0, 1], [1, 0], lw=2, c='k', linestyle='--')
            dx.plot([0, 1], [0, 1], lw=2, c='k', linestyle='--')

            bx.legend(fontsize = 15)
            dx.legend(fontsize = 15)

            #==============================================================
            # Save each optimization result as a CSV file
            result_df.to_csv(os.path.join(path, f'{opt}_table5 {model_name} stage.csv'), index=False)

        # Concatenate all results  
        final_df = pd.concat(all_df, axis=0, ignore_index=True)
        final_df.to_csv(os.path.join(path, f'table5 {model_name} stage.csv'), index=False)

        criteria.Heatmap(final_df, [f'original {model_name}', f"LASSO {model_name}", f"ANOVA {model_name}", f"GA {model_name}", f"DE {model_name}", 
                                    f"PSO {model_name}", f"WOA {model_name}", f"HHO {model_name}", f"SMA {model_name}", f"FGOA {model_name}"])
        print(final_df)
        plt.show()

        #==============================================================
        # SHAP
        if model_name =="A-LSTM":
            trained_model = load_model(f'C:\\Users\\user\\OneDrive - 國立高雄科技大學\\文件\\GitHub\\FGOA-ccRCC\\model\\FGOA A-LSTM.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
            explainer = shap.GradientExplainer (trained_model, X_train)
            shap_values = explainer.shap_values(X_test)
            print(shap_values.shape)
            shap_values = shap_values[:,-1,:]
            print(shap_values.shape)

            shap_values_reshaped = shap_values.reshape(X_test_2d.shape[0], X_test_2d.shape[1])
            print(shap_values_reshaped.shape)

            shap.summary_plot(shap_values_reshaped, X_test_2d, feature_names=feature_names)

            plt.show()