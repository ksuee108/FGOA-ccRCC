from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier as LGBM
from catboost import CatBoostClassifier
import numpy as np
import shap
import matplotlib.pyplot as plt
class ML:
    def __init__(self, X_train, X_test, y_train, y_test, tree_depth, tree_num, learning_rate,model_name, *args, **kwargs):
        """
        這個類封裝了多種機器學習分類算法的訓練和測試過程。
        包括 RF, SVM, KNN, GDBT, XGB, CatBoost, LGBM

        初始化 ML 類並設置訓練和測試數據。

        :param X_train: 訓練集特徵數據 (2D 數組)
        :param X_test: 測試集特徵數據 (2D 數組)
        :param y_train: 訓練集標籤 (1D 數組)
        :param y_test: 測試集標籤 (1D 數組)
        :param tree_depth: 樹模型的深度 (整數)
        :param tree_num: 樹模型的數量 (整數)
        :param learning_rate: 學習率 (浮點數)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.tree_depth = tree_depth
        self.tree_num = tree_num
        self.learning_rate = learning_rate
        self.model_name = model_name
    def RF_ROC(self):
        """
        使用隨機森林分類器訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        RF = RFC(max_depth=self.tree_depth, n_estimators=self.tree_num)
        RF.fit(self.X_train, self.y_train)
        y_pred = RF.predict(self.X_test)
        y_pro = RF.predict_proba(self.X_test)[:, 1]

        explainer = shap.Explainer(RF.predict_proba, self.X_train)
        #shap_values = explainer.shap_values(self.X_test)
        return y_pred, y_pro#, shap_values, explainer

    def SVM_ROC(self):
        """
        使用支持向量機訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        neigh = SVC(probability=True)
        neigh.fit(self.X_train, self.y_train)
        y_pred = neigh.predict(self.X_test)
        y_pro = neigh.predict_proba(self.X_test)[:, 1]

        explainer = shap.Explainer(neigh.predict_proba, self.X_train)  # 訓練數據作為背景資料
        #shap_values = explainer(self.X_test)  # 獲取 SHAP 解釋結果
        return y_pred, y_pro#, shap_values, explainer

    def KNN_ROC(self):
        """
        使用 K 近鄰算法訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        neigh = KNN(n_neighbors=7)
        neigh.fit(self.X_train, self.y_train)
        y_pred = neigh.predict(self.X_test)
        y_pro = neigh.predict_proba(self.X_test)[:, 1]

        explainer = shap.Explainer(neigh.predict_proba, self.X_train)
        #shap_values = explainer.shap_values(self.X_test)
        return y_pred, y_pro#, shap_values, explainer

    def GBDT_ROC(self):
        """
        使用梯度提升樹訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        gbdt = GBDT(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num)
        gbdt.fit(self.X_train, self.y_train)
        y_pred = gbdt.predict(self.X_test)
        y_pro = gbdt.predict_proba(self.X_test)[:, 1]

        explainer = shap.Explainer(gbdt)
        #shap_values = explainer.shap_values(self.X_test)
        return y_pred, y_pro#, shap_values, explainer

    def XGB_ROC(self):
        """
        使用 XGBoost 訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        XGB_clf = XGBClassifier(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num, tree_method='gpu_hist')

        XGB_clf.fit(self.X_train, self.y_train)
        y_pred = XGB_clf.predict(self.X_test)
        y_prob = XGB_clf.predict_proba(self.X_test)
        probabilities = np.exp(y_prob)  # 对数概率转实际概率
        y_pro = probabilities[:, 1]

        explainer = shap.Explainer(XGB_clf)
        #shap_values = explainer.shap_values(self.X_test)
        return y_pred, y_pro#, shap_values, explainer

    def CatBoost_ROC(self):
        """
        使用 CatBoost 訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """

        model_multi = CatBoostClassifier(depth=self.tree_depth, learning_rate=self.learning_rate, iterations=self.tree_num, task_type='GPU')

        model_multi.fit(self.X_train, self.y_train)
        y_pred = model_multi.predict(self.X_test)
        y_prob = model_multi.predict_log_proba(self.X_test)
        probabilities = np.exp(y_prob)  # 对数概率转实际概率
        y_pro = probabilities[:, 1]

        explainer = shap.Explainer(model_multi)
        #shap_values = explainer.shap_values(self.X_test)
        return y_pred, y_pro#, shap_values, explainer
    #LGBM
    def LGBM_ROC(self):
        """
        使用 LightGBM 訓練模型並返回預測結果。

        :return: 預測結果 (1D 數組)
        """
        model_multi = LGBM(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num, device='gpu')

        model_multi.fit(self.X_train, self.y_train)
        y_pred = model_multi.predict(self.X_test)

        y_pro = model_multi.predict_proba(self.X_test)[:, 1]
        
        explainer = shap.Explainer(model_multi)
        #shap_values = explainer(self.X_test)
        return y_pred, y_pro#, shap_values, explainer