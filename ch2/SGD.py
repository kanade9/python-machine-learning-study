# p45 確率的勾配降下法 SGD の実装 (ADALINEを改良する)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ch2.correct import plot_decision_regions


class AdalineSGD(object):
    """param
    eta: float 学習率 0.0から1まで
    n_iter : int トレーニングデータのトレーニング回数
    random_state : int 重み初期化のための乱数シード

    属性
    w_ 一次元の配列 適合後の重みを表す
    cost_ list 各エポックでの誤差平方和のコスト関数
    """

    # shuffle追加とrandom_stateをNoneに変更
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter

        # 重みの初期化フラグを追加。Falseに設定しておく
        self.w_initialized = False
        # 各エポックでトレーニングデータをシャッフルするかのフラグを初期化する
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: shape=[n_samples, n_features],[サンプルの個数,特徴量の個数]
        :param Y: shape=[n_samples]
        :return: self obj
        """

        # 重みベクトルの生成
        self.initialize_weights(X.shape[1])
        self.cost_ = []
        # トレーニング回数だけ反復
        for i in range(self.n_iter):
            # 指定した場合にトレーニングデータをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)

            # 各サンプルのコストを格納するリストを生成する
            cost = []
            # 各サンプルごとに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重み更新とコストの計算
                cost.append(self._update_weights(xi, target))

            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """重みを再初期化することなくトレーニングデータに適合させる"""

        # 初期化されてなかったら初期化するよ
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])

        """目的変数yの要素が2以上の時には
        各サンプルの特徴量xiと目的変数targetで重みを更新する"""
        if y.ravel().shape[1] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)

        else:
            self._update_weights(X, y)
        """目的変数yの要素数が1の場合は
                サンプル全体の特徴量Xと目的変数yで重みを更新する"""

        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        """重みを小さな乱数に初期化する"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ADALINEで重みを更新する"""
        # 活性化関数の出力の計算をする
        output = self.activation(self.net_input(xi))
        # 誤差計算
        error = (target - output)
        # 重み更新
        # si.dot(error)の部分が理解できていない。
        self.w_[1:] += self.eta * xi.dot(error)
        # 重み更新(w0)
        self.w_[0] += self.eta * error
        # コスト計算
        cost = 0.5 * error ** 2

        return cost

    def net_input(self, X):
        # 総入力の計算
        return np.dot(X, self.w_[1:], ) + self.w_[0]

    def activation(self, X):
        # 活性化関数の出力はここに書く
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# ここから学習して表示させるコード
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X_std = df.iloc[0:100, [0, 2]].values

# 学習
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# エポックと折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
