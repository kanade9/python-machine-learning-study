#!/usr/bin/env python
# coding: utf-8

# In[2]:


# pycharmでjupyter起動できる？？
print("hello")

# In[5]:


# pandasテスト　画像の表示
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

# In[12]:


import matplotlib.pyplot as plt, numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setisa', -1, -1)
# 1列、3列の抽出(1列目はがく辺の長さ、3列目は花びらの長さ)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

# なんでXの50~100はversicolorになるんだ？？データセットが綺麗に整理されているから？？
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例を配置
plt.legend(loc='upper left')
plt.show()

# In[17]:


# 2-2.pyで作ったパーセプトロンを使って分類してみるよ
# jupyterで他のpythonファイルをimportする
# http://romth.hatenablog.com/entry/2016/12/08/214641http://romth.hatenablog.com/entry/2016/12/08/214641
# 特別な設定をしなくてもインポートできた。
from ch2.perceptron import Perceptron

ppn = Perceptron(eta=0.01, n_iter=10)
ppn.fit(X, y)
# 折れ線グラフのプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of update')

plt.show()
#
# エラーは出ないけど線が一直線で正しく描画されなかった。

# In[16]:


import subprocess

subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '2-3.ipynb'])
