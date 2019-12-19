
"""
zip関数の使い方。複数のループをまとめる時などに使う
要素数が一致しない場合は”多い方の”要素が無視される。（少ない方の個数にループの回数が設定される）
    names = ['Alice', 'Bob', 'Charlie']
    ages = [24, 50, 18]
    for name, age in zip(names, ages):
        print(name, age)

        ------------
        Alice 24
        Bob 50
        Charlie 18
        -------------
"""
"""
np.whereの使い方
import numpy as np

a = np.arange(9).reshape((3, 3))
print(a)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

4以下のものには-1、それ以外の部分では100にするndarrayを返す
print(np.where(a < 4, -1, 100))
# [[ -1  -1  -1]
#  [ -1 100 100]
#  [100 100 100]]

np.arangeの使い方。float型を扱うこともできる
print(np.arange(3, 10, 2))
# [3 5 7 9]

よく忘れるenumerate。
iを使って順番を出力するのに用いられている。
for i, name in enumerate(l):
    print(i, name)
# 0 Alice
# 1 Bob
# 2 Charlie
"""