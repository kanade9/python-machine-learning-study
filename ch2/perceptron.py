# p25 パーセプトロンの実装
import numpy as np


# パーセプトロン分類機の作成

class Perceptron(object):
    """
    eta:　float 学習率　0<=eta<=1
    n_iter: int トレーニングデータのトレーニング回数
    random_state: int 重み初期化の乱数シード
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: shape= [n_samples(サンプルの個数) , n_features(特徴量の個数) ]　トレーニングデータ
        :param y:　shape= [n_samples]　目的変数
        :return:  self: Object
        """
        # RandomStateでシードの設定（固定もできる）
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # 重み更新
            for xi, target in zip(X, y):
                # 重みw1~wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重みw0の更新
                self.w_[0] += update
                # 重み更新が0でないときは誤分類としてカウントする？？？？意味がわからん
                # 　重み更新が0のとき適するデータになるので、0でないということは適してない結果だからエラーに加算する
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # 　総入力を計算する
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, -1)

