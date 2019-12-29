import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.cross_validationは書籍のミス??存在しない。
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
