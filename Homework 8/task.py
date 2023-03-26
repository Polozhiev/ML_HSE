from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, count=np.unique(x,return_counts=True)
    return np.sum(count/len(x) * (1-count/len(x)))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, count=np.unique(x,return_counts=True)
    return -np.sum(count/len(x) * np.log2(count/len(x)))



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
    union=np.concatenate((left_y, right_y))
    return criterion(union)-len(left_y)/len(union)*criterion(left_y)-len(right_y)/len(union)*criterion(right_y)

# Task 2

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
        self.prob_dict=dict(zip(values, counts/len(ys)))
class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini", 
                 max_depth : Optional[int] = None, 
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion=gini
        if criterion=="entropy":
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf 
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.X=X
        self.y=y
        self.root=self.build(X,y,depth=0)


    def build(self,X,y,depth):
        if len(X)<=self.min_samples_leaf or len(np.unique(y))==1:
            return DecisionTreeLeaf(y)
        if self.max_depth:
            if depth==self.max_depth:
                return DecisionTreeLeaf(y)

        max_gain=-1
        for dim in range(X.shape[1]):
            for value in np.unique(X[:,dim]):
                left=np.where(X[:,dim]<value)
                right=np.where(X[:,dim]>=value)
                curr_gain=gain(y[left],y[right],self.criterion)
                _min_samples_leaf = min(len(y[left]), len(y[right]))
                if curr_gain>max_gain and _min_samples_leaf >= self.min_samples_leaf:
                    max_gain=curr_gain
                    split_dim=dim
                    split_value=value
                    best_left=left
                    best_right=right
        if max_gain==-1:
            return DecisionTreeLeaf(y)    
        left_tree=self.build(X[best_left], y[best_left],depth+1)
        right_tree=self.build(X[best_right], y[best_right], depth+1)
        return DecisionTreeNode(split_dim,split_value,left_tree, right_tree)


    
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        list_dict_prob=[]
        for x in X:
            node=self.root
            while isinstance(node,DecisionTreeNode):
                if x[node.split_dim]>node.split_value:
                    node=node.right
                else:
                    node=node.left
            list_dict_prob.append(node.prob_dict)
        return list_dict_prob
    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4

task4_dtc=DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=1)