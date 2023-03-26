import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn

from sklearn.metrics import accuracy_score


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations=iterations
    
    def transform_labels(self, y):
        self.labels=np.unique(y)
        y_new=np.ones(len(y))
        y_new[y==self.labels[0]]=-1
        return y_new.astype('int')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        y_sign=self.transform_labels(y)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        w=np.zeros(X.shape[1])
        for _ in range(self.iterations):
            is_wrong=(np.sign(X@w)!=y_sign)
            w+=np.dot(X[is_wrong].T,y_sign[is_wrong])
        self.w=w
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y=np.sign(X@self.w).astype('int')
        y_old=np.zeros(len(y))
        y_old[y==-1]=self.labels[0]
        y_old[y==1]=self.labels[1]
        return y_old.astype('int')
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w=None
        self.iterations=iterations

    def transform_labels(self, y):
        self.labels=np.unique(y)
        y_new=np.ones(len(y))
        y_new[y==self.labels[0]]=-1
        return y_new.astype('int')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        y_true=self.transform_labels(y)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        w=np.zeros(X.shape[1])
        min_wrong=X.shape[0]
        for _ in range(self.iterations):
            y_pred=np.sign(X@w)
            is_wrong=(y_pred!=y_true)
            if np.sum(is_wrong)<min_wrong:
                min_wrong=np.sum(is_wrong)
                best_w=np.copy(w)
            w+=np.dot(y_true[is_wrong],X[is_wrong])

        self.w=best_w
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y=np.sign(X@self.w).astype('int')
        y_old=np.zeros(len(y))
        y_old[y==-1]=self.labels[0]
        y_old[y==1]=self.labels[1]
        return y_old.astype('int')
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    result = np.zeros((images.shape[0], 2))
    symmetry_hor = np.abs(images - images[:, ::-1]).mean(axis=(1, 2))
    symmetry_vert = np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
    result.T[0] = symmetry_hor
    result.T[1] = symmetry_vert

    return result
