import numpy as np
import pandas
import random
import copy
from typing import NoReturn
from collections import defaultdict

# Task 1

def cyclic_distance(points, dist):
    sum=0
    for i in range(len(points)):
        sum+=dist(points[i],points[(i+1)%len(points)])
    return sum

def l2_distance(p1, p2):
    return np.linalg.norm(p2-p1)

def l1_distance(p1, p2):
    return np.linalg.norm(p2-p1, ord=1)

# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist # Do not change
    
    def optimize(self, X):
        return self.optimize_explained(X)[-1]
    
    def optimize_explained(self, X):
        n=len(X)

        start_dist=np.inf
        for i in range(len(X)):
            tmp_path=np.random.permutation(n)
            tmp_dist=cyclic_distance(X[tmp_path], self.dist)
            if tmp_dist<start_dist:
                start_dist=tmp_dist
                curr_path=tmp_path
            
        paths=[curr_path]
        tmp_dist=cyclic_distance(X[curr_path], self.dist)
        for _ in range(self.max_iterations):
            found_better=False
            for i in range(n):
                for j in range(i+1,n):
                    new_path=curr_path.copy()
                    new_path[i], new_path[j] = new_path[j], new_path[i]
                    new_dist=cyclic_distance(X[new_path], self.dist)
                    if new_dist<tmp_dist:
                        tmp_dist=new_dist
                        curr_path=new_path
                        paths.append(new_path)
                        found_better=True
                        break
                else: 
                    continue
                break
            if not found_better:
                return paths

def synthetic_points(count=25, dims=2):
    return np.random.randint(40, size=(count, dims))

X = synthetic_points()
hc = HillClimb(100, l2_distance)


        

# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
    
    def optimize(self, X):
        return self.selection(self.optimize_explain(X)[-1])[0]
    
    def selection(self,population):
        # dists=[]
        # for ind in population:
        #     dists.append(cyclic_distance(self.X[ind],self.dist))
        # dists=np.array(dists)
        dists= np.array([cyclic_distance(X[creature], self.dist) for creature in population])
        best=dists.argsort()[:self.surv_size]
        return population[best]
    
    def crossover(self,first, second):
        m=len(first)
        i, j = np.sort(np.random.randint(low=0,high=m, size=2))
        new_ind = np.zeros(m, dtype=int)
        new_ind[0:(j - i)]=first[i:j]
        from_first=set(first[i:j])
        index = j - i
        for el in second:
            if el not in from_first:
                new_ind[index]=el
                index += 1
        return new_ind
        
    def reproduction(self, survivors):
        new_generation=[]
        for _ in range(self.pop_size):
            first,second = np.random.randint(low=0, high=len(survivors),size=2)
            new_generation.append(self.crossover(survivors[first], survivors[second]))
        return np.array(new_generation)

    def optimize_explain(self, X):
        self.X=X
        n = len(X)
        population = np.array([np.random.permutation(n) for _ in range(self.pop_size)])
        paths = [population]
        for _ in range(self.iters):
            survivors = self.selection(population)
            population = self.reproduction(survivors)
            paths.append(population)
        return paths

# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.X=X
        self.voc_limit=voc_limit
        d=defaultdict(int)
        for s in X:
            s=s.lower()
            s=s.split(" ")
            for w in s:
                d[w]+=1
        if len(d)>voc_limit:
            sorted_freq = sorted(d.items(), key=lambda x: x[1], reverse=True)
            self.dict=defaultdict()
            for i in range(voc_limit):
                self.dict[sorted_freq[i][0]]=sorted_freq[i][1]
            self.vocab_size=voc_limit
        else:
            self.dict=d
            self.vocab_size=len(d)

        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            который необходимо векторизовать.
        
        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        vectorize_matrix=np.zeros((len(X), self.vocab_size))
        list_keys=list(self.dict)
        for idx,s in enumerate(X):
            s=s.lower()
            s=s.split(" ")
            for w in s:
                if w in self.dict:
                    vectorize_matrix[idx][list_keys.index(w)]+=1
        return vectorize_matrix


# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        self.classes, y_counts=np.unique(y, return_counts=True)
        self.y_prob=y_counts/len(y)
        n_features=X.shape[1]
        self.prior_probs=np.zeros( (len(self.classes),n_features) )
        for i, y_value in enumerate(self.classes):
            for j in range(n_features):
                self.prior_probs[i,j]+=X[np.where(y==y_value)][j]+self.alpha
                self.prior_probs[i,j]/=np.sum(X[np.where(y==y_value)])+self.alpha*n_features
        
    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]
    
    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу. 
            Матрица размера (X.shape[0], n_classes)
        """
        log_probs=np.zeros( (X.shape[0], len(self.classes)) )
        log_probs+=np.log(self.y_prob)
        for class_index in range(len(self.classes)):
            log_probs[:, class_index] += (X * np.log(self.prior_probs[class_index, :])).sum(axis=0)

        
        return None