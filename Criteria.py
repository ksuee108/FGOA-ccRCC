import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score, precision_score,confusion_matrix,f1_score, precision_recall_curve
import seaborn as sns

class Evaluation_Criteria:

    def __init__(self, y_train, y_test, ax, bx, path):

        self.y_train = y_train
        self.y_test = y_test
        self.ax = ax
        self.bx = bx
        #self.fx = fx
        self.path = path
        
    #混淆矩陣
    def confusion(self, y_pred, predictions, x):

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
            "Gain":2*auc(fpr, tpr)-1
            }
        ch = pd.DataFrame(ch,index=[1])
        return ch
    
    @staticmethod
    #Heatmap
    def Heatmap(df,modle):
        sns.set(font_scale=1.5)
        fig3,cx = plt.subplots()
        plt.rcParams['font.family'] = 'Times New Roman'
        norm_cm = round(df[['TPR','FNR','FPR','TNR']].astype(float),3)
        transposed_norm_cm = norm_cm.transpose()

        cx = sns.heatmap(transposed_norm_cm, cmap='magma', vmin=0, vmax=1, annot=True, annot_kws={"size": 20, "fontname": "Times New Roman"}, square=True, 
                        xticklabels=modle, fmt=".3f")

        cx.set_xticklabels(cx.get_xticklabels(), fontfamily='Times New Roman', fontsize=20, rotation=45)
        cx.set_yticklabels(cx.get_yticklabels(), fontfamily='Times New Roman', fontsize=20)

        cx.xaxis.tick_top()
        cx.xaxis.set_label_position('top')
        cx.tick_params(axis='x', length=0)
        cx.tick_params(axis='y', length=0)

            
    #Two barChart
    @staticmethod
    def  Two_barChart(df):

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