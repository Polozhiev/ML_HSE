import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """

    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.normal(
            0, 2/np.sqrt(in_features+out_features), size=(in_features, out_features))
        self.b = np.random.normal(
            0, 2/np.sqrt(in_features+out_features), size=(out_features))
        self._x = None
        self.d = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        self._x = x
        return x@self.W+self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim < 2:
            d = d.reshape(1, -1)
        self.dW = self._x.T @ d
        self.db = d.sum(axis=0)
        return d @ self.W.T

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W = self.W-alpha*self.dW
        self.b = self.b-alpha*self.db


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        self._x = x
        return np.maximum(0, x)

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if d.ndim < 2:
            d = d.reshape(1, -1)
        return (self._x >= 0) * d


class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, x):
        e_x = np.exp(x - np.max(x))
        self._x = e_x / e_x.sum(axis=1, keepdims=True)
        return self._x

    def backward(self, y):
        t = np.eye(self._x.shape[1])[y]
        return self._x-t

# Task 2


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """

        if modules[-1] != Softmax():
            modules.append(Softmax())
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        for _ in range(self.epochs):
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            for i in range(0, len(X), self.batch_size):
                X_i = X[i:(i+self.batch_size)]
                y_i = y[i:(i+self.batch_size)]
                self.predict_proba(X_i)
                d = self.modules[-1].backward(y_i)
                for m in reversed(self.modules[:-1]):
                    d = m.backward(d)
                for m in self.modules[:-1]:
                    m.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        for m in self.modules:
            X = m.forward(X)
        return X

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

# Task 3


classifier_moons = MLPClassifier([Linear(2, 64),
                                  ReLU(),
                                  Linear(64, 64),
                                  ReLU(),
                                  Linear(64, 2)])  # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier([Linear(2, 64),
                                  ReLU(),
                                  Linear(64, 64),
                                  ReLU(),
                                  Linear(64, 3)])  # Нужно указать гиперпараметры

# Task 4


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        self.load_state_dict(torch.load(__file__[:-7] + "/model.pth",map_location=torch.device('cpu')))

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        torch.save(self.state_dict(), "model.pth")


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    predicted = model(X)
    return F.cross_entropy(predicted, y)
