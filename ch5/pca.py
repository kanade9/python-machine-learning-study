import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.cross_validationは書籍のミス??存在しない。
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
                      , header=None)

# 2列め[以降]のデータをXに、1列目のデータをyに格納する
# pandasのilocでは行番号、列番号を0基準で指定する
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# df_wine.iloc[:,0:]にしててミスってた

# データをトレーニングデータとテストデータに分割
# stratifyはtestに正解ラベルを設定しておくとデータの値の比率が一致して(正解と不正解が)分割される。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# 標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 共分散行列の生成
cov_mat = np.cov(X_train_std.T)
# 固有値、固有ベクトルの計算
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
# print('\nEigenvecs \n%s' % eigen_vecs)

# 5.1.3 固有値の合計に対する固有値λjの割合（分散説明率）の描画
tot = sum(eigen_vals)
# 分散説明率の計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積わをcumsumで計算。(memo参照)
cum_var_exp = np.cumsum(var_exp)

# 分散説明率棒グラフ生成
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')

# 分散説明率の累積和の階段グラフ生成
plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')

plt.xlabel('Explained variance ratio')
plt.ylabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 5.1.4 特徴変換

# 固有値の大きいものから順に固有対をならべかえる
# (固有値、固有ベクトル)のタプルのリストを生成

# 固有値問題では絶対値をとってその大きさで比べるので、絶対値をとる。(np.abs)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# (固有値、固有ベクトル)のタプルを大きいものから順にならべかえる
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# print(eigen_pairs)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# 次はp147~
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

# クラスラベル、点の色、点の種類の組み合わせからなるリストを生成してプロットする
# lと1の違いに注意する！！
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c,
                label=1, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# 5.1.5 scikit-learnの主成分分析

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # 予測結果の元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.6, c=cmap(idx),
                    edgecolor='black', marker=markers[idx], label=cl)


pca = PCA(n_components=2)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# トレーニングデータでロジスティック回帰を実行
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
