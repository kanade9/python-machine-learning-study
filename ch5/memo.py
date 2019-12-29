"""
# ndarrayの累積和を求めるcumsum
a = np.array([1,2,3,4,5,6])
#下記どちらの書き方でもOK
np.cumsum(a)
a.cumsum()
> array([ 1,  3,  6, 10, 15, 21])
> array([ 1,  3,  6, 10, 15, 21])
"""