# p52 ch3 scikit-learnの活用
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
