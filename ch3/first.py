# p52 ch3 scikit-learnの活用
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# 3,4列目の特徴量を抽出
# 特徴量は整数ラベルで表すことが慣例となっている(文字ラベルは使わない)
X = iris.data[:, [2, 3]]
# クラスラベルの取得
y = iris.target

# 一意なクラスラベルの出力
print('Class labels:', np.unique(y))

# 全体データの30%をテストデータに分類
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# train_test_splitでデータを勝手にシャッフルしてくれる。
# stratifyで層化サンプリング？？トレーニングとテストに入っているクラスラベル比率が入力データセットと同じくなる。以下確認
print('Labels counts in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 次はp55~
# パーセプトロンインスタンス作成

# TypeError: __init__() got an unexpected keyword argument 'n_iter'
# versionエラー??
# n_iterをmax_iterに変更することで対処可能。
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
# 誤分類の表示
# max_iterがサンプルと異なる。max_iterにしてるのが原因??
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# 分類の正解率を表示
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# p56 決定領域のプロットでどの程度識別できるかの可視化
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # マーカーとカラーマップ準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のplot
    # -1しているのはindexをそろえるためでいいのかな？？
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイント生成
    # np.arrangeでx1_minからx1_maxまでで,間隔がresolutionになるndarrayを生成。
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

    # ここが改良部分！！
    #  テストサンプルを目立たせるため点をoで表示
    if test_idx:
        # 全てサンプルをプロット
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolors='black', alpha=1.0,
                    linewidths=1, marker='o', s=100, label='test set')


# トレーニングデータ、テストデータも特徴量を行方向に結合する
X_combined_std = np.vstack((X_train_std, X_test_std))
# トレーニングデータとテストデータのクラスラベル結合
y_combined = np.hstack((y_train, y_test))
# 決定境界プロット
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 次はp58のロジスティック回帰から
