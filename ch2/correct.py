# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップ準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のplot
    # -1しているのはindexをそろえるためでいいのかな？？
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイント生成
    # np.arangeでx1_minからx1_maxまでで,間隔がresolutionになるndarrayを生成。
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # 予測結果を元のグリッドポイントのデータサイズに変換
    # クラスラベルを予測したあとはxx1,xx2と同じ次元をもつグリッドに作り変えなければならない。
    # なぜ？？？
    Z = Z.reshape(xx1.shape)

    # グリッドポイントの等高線をプロットする
    # contourfのプロットの仕方よくわからん！とりあえず図を塗ってくれることはわかった。
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロットする
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolors='black')


plot_decision_regions(X, y, classifier=ppn)
# 軸ラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# この次はADALINE

class AdalineGD(object):
    """param
    eta: float 学習率 0.0から1まで
    n_iter : int トレーニングデータのトレーニング回数
    random_state : int 重み初期化のための乱数シード

    属性
    w_ 一次元の配列 適合後の重みを表す
    cost_ list 各エポックでの誤差平方和のコスト関数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: shape=[n_samples, n_features],[サンプルの個数,特徴量の個数]
        :param Y: shape=[n_samples]
        :return: self obj
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        self.cost_ = []
        # トレーニング回数だけ反復
        for i in range(self.n_iter):
            net_input = self.net_input(X)

            # ここは後ほどの実装で使うとこ。今は特に何もしないよ
            output = self.activation(net_input)
            # 誤差の計算をする
            errors = (y - output)

            # ここなぜ転置する必要がある？？ ⇨特徴行列と誤差ベクトルの行列ベクトル積をかけてるんだけど、、、
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # 　コスト関数の計算
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        # 総入力の計算
        return np.dot(X, self.w_[1:], ) + self.w_[0]

    def activation(self, X):
        # 活性化関数の出力はここに書く
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# 描画領域を分割するよ
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# ADALINE学習
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

# 学習率 0.01の場合の図作成
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# 学習率 0.0001の場合の図作成
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()

# 2-5-2 スケーリングの手法"標準化"を用いて勾配効果法を改善する

# これはdeepcopy
X_std = np.copy(X)
# 各列の標準化 .stdで標準偏差を求めることができる
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
# 描画
plot_decision_regions(X_std, y, classifier=ada)

plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 凡例
plt.legend(loc='upper left')

# グラフを画像からはみ出さないようにサイズを調整して表示
plt.tight_layout()
plt.show()

# エポック数とコストの関係グラフ
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
plt.show()
