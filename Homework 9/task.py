from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# Task 0


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, count=np.unique(x,return_counts=True)
    return np.sum(count/len(x) * (1-count/len(x)))
    
    probs = np.bincount(x) / len(x)
    return 1 - np.sum(probs ** 2)


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, count=np.unique(x,return_counts=True)
    return -np.sum(count/len(x) * np.log2(count/len(x)))

    probs = np.bincount(x) / len(x)
    return -np.sum(probs * np.log2(probs + 1e-16))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    union = np.concatenate((left_y, right_y))
    return criterion(union)-len(left_y)/len(union)*criterion(left_y)-len(right_y)/len(union)*criterion(right_y)


# Task 1
class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        self.y = None
        values, counts = np.unique(ys, return_counts=True)
        self.y = values[np.argmax(counts)]
      #  self.prob_dict=dict(zip(values, counts/len(ys)))


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """

    def __init__(self, split_dim: int,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.root = None
        if criterion == "entropy":
            self.criterion = entropy
        else:
            self.criterion = gini
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        if max_features == "auto":
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            self.max_features = max_features
#
        # old_class = np.unique(y)
        # new_class = np.arange(len(old_class))
        # self.cls_dct = dict(zip(old_class, new_class))
        # self.rev_cls_dct = dict(zip(new_class, old_class))
        # y = np.array([self.cls_dct[i] for i in y])
#
        self.train_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        self.oob_idx = np.setdiff1d(np.arange(len(X)), self.train_idx)
        self.X_oob=X[self.oob_idx]
        self.y_oob=y[self.oob_idx]
        self.root = self.build(X[self.train_idx], y[self.train_idx], depth=0)

    def build(self, X, y, depth):
        if len(X) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            return DecisionTreeLeaf(y)
        if self.max_depth and depth == self.max_depth:
            return DecisionTreeLeaf(y)

        max_gain = -1
        features = np.random.choice(X.shape[1], self.max_features, replace=False)
        for dim in features:
            left = np.where(X[:, dim] == 0)
            right = np.where(X[:, dim] == 1)
            curr_gain = gain(y[left], y[right], self.criterion)
            if (curr_gain > max_gain) and ( min(len(y[left]), len(y[right])) ) >= self.min_samples_leaf:
                max_gain = curr_gain
                split_dim = dim
                best_left = left
                best_right = right
        if max_gain == -1:
            return DecisionTreeLeaf(y)
        left_tree = self.build(X[best_left], y[best_left], depth+1)
        right_tree = self.build(X[best_right], y[best_right], depth+1)
        return DecisionTreeNode(split_dim, left_tree, right_tree)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        list_dict_prob = []
        for x in X:
            node = self.root
            while isinstance(node, DecisionTreeNode):
                if x[node.split_dim] == 1:
                    node = node.right
                else:
                    node = node.left
            list_dict_prob.append(node.y)
        return np.array(list_dict_prob)

# Task 2


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest = None

    def fit(self, X, y):
        self.forest = []
        self.X=X
        self.y=y
        for _ in range(self.n_estimators):
            tree = DecisionTree(X, y, self.criterion, self.max_depth, self.min_samples_leaf, self.max_features)
            self.forest.append(tree)
        
    def predict(self, X):
        # predicts = self.forest[0].predict(X)
        # for tree in self.forest[1:]:
        #     predicts = np.vstack((predicts, tree.predict(X)))
        # predict_final = mode(predicts, axis=0)
        # return predict_final.mode.ravel()
        
        
        type_of_data=self.forest[0].predict(X).dtype
        predicts = np.zeros((X.shape[0], self.n_estimators),dtype=np.dtype(type_of_data))
        for i in range(self.n_estimators):
            predicts[:, i] = self.forest[i].predict(X)

        return mode(predicts, axis=1)[0].reshape(-1)


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)

# X, y = synthetic_dataset(10)

# rfc = RandomForestClassifier(n_estimators=3)
# rfc.fit(X, y)


# acc=[]
# for i in range(100):
#     X, y = synthetic_dataset(1000)
#     rfc = RandomForestClassifier(n_estimators=5)
#     rfc.fit(X, y)
#     acc.append(np.mean(rfc.predict(X) == y))

# Task 3

def out_of_bag_error(rfc, X,y):
    true=0
    n=0
    for i in range(len(X)):
        pred=[]
        for tree in rfc.forest:
            if i in tree.oob_idx:
                pred.append(tree.predict(X[i])[0])
        if pred:
            pred_mode=mode(pred).mode[0]
            if pred_mode==y[i]:
                true+=1
        n+=1
    return 1-true/n 

def feature_importance(rfc):
    err_oob=out_of_bag_error(rfc, rfc.X, rfc.y)
    importance=[]
    for j in range(rfc.X.shape[1]):
        X_j=rfc.X.copy()
        np.random.shuffle(X_j[:, j])
        err_oob_j=out_of_bag_error(rfc,X_j,rfc.y)
        importance.append(err_oob_j - err_oob)
    return importance

def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]   
 

# Task 4

rfc_age = RandomForestClassifier(n_estimators=10, max_depth=20, max_features=5, criterion="gini")
rfc_gender = RandomForestClassifier(n_estimators=5, max_depth=10, max_features=5, criterion="gini")

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model

# cat_age=CatBoostClassifier(iterations=100,loss_function='MultiClass')
# cat_age.fit(X_train,y_age_train,verbose=False)
# cat_age.save_model('cat_age')

# cat_sex=CatBoostClassifier(iterations=100,loss_function='MultiClass')
# cat_sex.fit(X_train,y_sex_train,verbose=False)
# cat_sex.save_model('cat_sex')

catboost_rfc_age = CatBoostClassifier(iterations=100,loss_function='MultiClass')
catboost_rfc_age.load_model(__file__[:-7] +'/cat_age.cbm')
catboost_rfc_gender = CatBoostClassifier(iterations=100,loss_function='MultiClass')
catboost_rfc_age.load_model(__file__[:-7] +'/cat_sex.cbm')
