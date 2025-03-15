import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score, precision_score,confusion_matrix,f1_score, precision_recall_curve
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

class Evaluation_Criteria:
    """
    這個類提供了一系列評估機器學習模型的功能，包括混淆矩陣、熱力圖、條形圖等。

    屬性：
        y_train(array-like):訓練集標籤。
        y_test(array-like):測試集標籤。
        ax(matplotlib.Axes):用於繪製 ROC 曲線的坐標軸。
        bx(matplotlib.Axes):用於繪製 Precision-Recall 曲線的坐標軸。
    """
    def __init__(self, y_train, y_test, ax, bx, path):
        """
        初始化 Evaluation_Criteria 類。

        參數：
            y_train(array-like):訓練集標籤。
            y_test(array-like):測試集標籤。
            ax(matplotlib.Axes):用於繪製 ROC 曲線的坐標軸。
            bx(matplotlib.Axes):用於繪製 Precision-Recall 曲線的坐標軸。
        """

        self.y_train = y_train
        self.y_test = y_test
        self.ax = ax
        self.bx = bx
        #self.fx = fx
        self.path = path
        
    #混淆矩陣
    def confusion(self, y_pred, predictions, x):
        """
        计算并返回混淆矩阵及相关评估指标。

        参数：
            y_pred(array-like): 预测结果。
            predictions(array-like): 模型的预测概率。
            x(str): 模型的名称。

        返回：
            pandas.DataFrame： 各种评估指标，包括准确度、精确度、召回率等。
        """
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        Acc = accuracy_score(self.y_test, y_pred)
        Precision = precision_score(self.y_test, y_pred)
        Sensitivity = recall_score(self.y_test, y_pred)
        Specificity = (tn / (fp + tn))
        F1 = f1_score(self.y_test, y_pred)
        TPR = (tp / (tp + fn))
        FPR = (fp / (tn + fp))
        TNR = (tn / (fp + tn))
        FNR = (fn / (tp + fn))

        fpr, tpr, _ = roc_curve(self.y_test, predictions)
        precision, recall, _ = precision_recall_curve(self.y_test, predictions)
        self.bx.plot(fpr, tpr, lw=2, label=f'{x} AUC area={auc(fpr, tpr):.3}' ) 
        self.ax.plot(recall, precision, lw=2, label=f'{x} AUC area={(auc(recall, precision)):.3}' )
        #self.ax.set_xlabel(fontsize=20)
        """if x == 'type 1' or x == 'type 2' or 'LGBM' in x or 'Catboost' in x or 'XGB' in x or 'RF' in x or 'KNN' in x or 'RSF' in x:
            correlation, p_value = pearsonr(np.array(self.y_test),  y_pred)
        else:
            correlation, p_value = pearsonr(np.array(self.y_test), np.array([item[0] for item in y_pred]))"""
        ch = {
            "Model":x,
            "Accuracy": Acc,
            "Precision": Precision,
            "Sensitivity": Sensitivity,
            "Specificity": Specificity,
            "TPR": TPR,
            "FNR": FNR,
            "FPR": FPR,
            "TNR": TNR,
            "F1 score": F1,
            """"Pearsonr correlation":correlation,
            "P vale":p_value,"""
            "Gain":2*auc(fpr, tpr)-1
            }
        ch = pd.DataFrame(ch,index=[1])
        return ch
    
    @staticmethod
    #Heatmap
    def Heatmap(df,modle):
        """
        生成並保存熱力圖。

        參數：
            df(pandas.DataFrame):包含評估指標的數據。
            modle(str列表):模型的名稱列表。
        """
        sns.set(font_scale=1.5)  # 全局字體比例，可以根據需要調整
        fig3,cx = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        norm_cm = round(df[['TPR','FNR','FPR','TNR']].astype(float),3)
        transposed_norm_cm = norm_cm.transpose()

        cx = sns.heatmap(transposed_norm_cm, cmap='magma', vmin=0, vmax=1, annot=True, annot_kws={"size": 20, "fontname": "Times New Roman"}, square=True, 
                        xticklabels=modle, fmt=".3f")

        # 調整 x 和 y 軸標籤字體
        cx.set_xticklabels(cx.get_xticklabels(), fontfamily='Times New Roman', fontsize=20, rotation=45)
        cx.set_yticklabels(cx.get_yticklabels(), fontfamily='Times New Roman', fontsize=20)

        # 將 x 軸標籤放在圖頂
        cx.xaxis.tick_top()
        cx.xaxis.set_label_position('top')
        cx.tick_params(axis='x', length=0)  # 移除 x 軸刻度
        cx.tick_params(axis='y', length=0)  # 移除 y 軸刻度

            
    #Two barChart
    @staticmethod
    def  Two_barChart(df):
        """
        繪製兩個條形圖，包含不同指標。

        參數：
            df(pandas.DataFrame):包含評估指標的數據。
        """
        fig4,dx = plt.subplots()
        df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Sensitivity', 'Specificity'], var_name='Metric', value_name='Percentage')
        # Plot the grouped bar chart
        sns.barplot(x="Percentage", y="Model", hue="Metric", data=df_melted, palette=["b", "orange"])
        # Add a legend and informative axis label
        dx.legend(ncol=2, loc="lower right", frameon=True)
        dx.set(xlim=(0, 1), xlabel="Percentage", ylabel="Model")
        dx.rcParams['font.family'] = 'Times New Roman'
    
    #comblie_barchart
    @staticmethod
    def Comblie_Barchart(df, modle):
        """
        繪製組合條形圖，顯示準確度和錯誤率的比例。

        參數：
            df(pandas.DataFrame):包含評估指標的數據。
            modle(str列表):模型的名稱列表。
        """
        fig4,ex = plt.subplots()
        CB = df.copy()
        CB = {'Model': modle,
                'Accuracy': df["Accuracy"],
                'Error Rate':  1 - df["Accuracy"] }
        df = pd.DataFrame(CB)
        df['Dataset1_percent'] = (df['Accuracy'] / 1) * 100
        df['Dataset2_percent'] = (df['Error Rate'] / 1) * 100
        ex.barh(df['Model'], df['Dataset1_percent'], label='Accuracy', color='lightblue')
        ex.barh(df['Model'], df['Dataset2_percent'], left=df['Dataset1_percent'], label='Error Rate', color='gold')
        for index, value in enumerate(df['Dataset1_percent']):
            plt.text(value / 2, index, f'{value:.2f}%', ha='center', va='center', color='black')

        for index, value in enumerate(df['Dataset2_percent']):
            plt.text(value / 2 + df['Dataset1_percent'][index], index, f'{value:.2f}%', ha='center', va='center', color='black')

        ex.set_xlabel('Percentage')
        ex.set_ylabel('Model')
        ex.set_title('Stacked Horizontal Bar Chart')
        plt.legend(loc='lower left')
        ex.rcParams['font.family'] = 'Times New Roman'