from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier as LGBM
from catboost import CatBoostClassifier
import numpy as np

class ML:
    def __init__(self, X_train, X_test, y_train, y_test, tree_depth, tree_num, learning_rate, model_name, *args, **kwargs):
        """
        This class encapsulates the training and testing process of multiple machine learning classification algorithms.
        Including RF, SVM, KNN, GDBT, XGB, CatBoost, LGBM

        Initialize the ML class and set the training and testing data.

        :param X_train: Training set feature data (2D array)
        :param X_test: Test set feature data (2D array)
        :param y_train: Training set labels (1D array)
        :param y_test: Test set labels (1D array)
        :param tree_depth: Depth of the tree model (integer)
        :param tree_num: Number of trees in the model (integer)
        :param learning_rate: Learning rate (float)
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
        Train the model using Random Forest Classifier and return the prediction results.

        :return: Prediction results (1D array)
        """
        RF = RFC(max_depth=self.tree_depth, n_estimators=self.tree_num)
        RF.fit(self.X_train, self.y_train)
        y_pred = RF.predict(self.X_test)
        y_pro = RF.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 

    def SVM_ROC(self):
        """
        Train the model using Support Vector Machine and return the prediction results.

        :return: Prediction results (1D array)
        """
        neigh = SVC(probability=True)
        neigh.fit(self.X_train, self.y_train)
        y_pred = neigh.predict(self.X_test)
        y_pro = neigh.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 

    def KNN_ROC(self):
        """
        Train the model using K-Nearest Neighbors algorithm and return the prediction results.

        :return: Prediction results (1D array)
        """
        neigh = KNN(n_neighbors=7)
        neigh.fit(self.X_train, self.y_train)
        y_pred = neigh.predict(self.X_test)
        y_pro = neigh.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 

    def GBDT_ROC(self):
        """
        Train the model using Gradient Boosting Decision Trees and return the prediction results.

        :return: Prediction results (1D array)
        """
        gbdt = GBDT(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num)
        gbdt.fit(self.X_train, self.y_train)
        y_pred = gbdt.predict(self.X_test)
        y_pro = gbdt.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 

    def XGB_ROC(self):
        """
        Train the model using XGBoost and return the prediction results.

        :return: Prediction results (1D array)
        """
        XGB_clf = XGBClassifier(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num, tree_method='gpu_hist')

        XGB_clf.fit(self.X_train, self.y_train)
        y_pred = XGB_clf.predict(self.X_test)
        y_prob = XGB_clf.predict_proba(self.X_test)
        probabilities = np.exp(y_prob)  # Convert log probabilities to actual probabilities
        y_pro = probabilities[:, 1]

        return y_pred, y_pro 

    def CatBoost_ROC(self):
        """
        Train the model using CatBoost and return the prediction results.

        :return: Prediction results (1D array)
        """
        model_multi = CatBoostClassifier(depth=self.tree_depth, learning_rate=self.learning_rate, iterations=self.tree_num, task_type='GPU')

        model_multi.fit(self.X_train, self.y_train)
        y_pred = model_multi.predict(self.X_test)
        y_prob = model_multi.predict_log_proba(self.X_test)
        probabilities = np.exp(y_prob)  # Convert log probabilities to actual probabilities
        y_pro = probabilities[:, 1]

        return y_pred, y_pro 

    def LGBM_ROC(self):
        """
        Train the model using LightGBM and return the prediction results.

        :return: Prediction results (1D array)
        """
        model_multi = LGBM(max_depth=self.tree_depth, learning_rate=self.learning_rate, n_estimators=self.tree_num, device='gpu')

        model_multi.fit(self.X_train, self.y_train)
        y_pred = model_multi.predict(self.X_test)

        y_pro = model_multi.predict_proba(self.X_test)[:, 1]

        return y_pred, y_pro 