import numpy as np


class SimpleLinearRegression:

    def __init__(self):
        '''初始化Simple Linear Regression模型'''
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''根据训练数据集训练模型'''
        assert x_train.ndim == 1, ''
        assert len(x_train) == len(y_train), ''

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        '''给定待预测数据集，返回预测结果'''
        assert x_predict.ndim == 1, ''
        assert self.a_ is not None and self.b_ is not None, ''

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个预测数据，返回预测结果值'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearRegression()'
