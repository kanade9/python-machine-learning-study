# vstackは配列を縦に連結、hstackは横に連結する
"""
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# -----------------------
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])
# -----------------------

b = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
# -----------------------
# array([[10, 20, 30],
#        [40, 50, 60],
#        [70, 80, 90]])
# -----------------------

np.vstack((a, b))
# -----------------------
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 20, 30],
#        [40, 50, 60],
#        [70, 80, 90]])
# -----------------------

np.hstack((a, b))
# -----------------------------------
# array([[ 1,  2,  3, 10, 20, 30],
#        [ 4,  5,  6, 40, 50, 60],
#        [ 7,  8,  9, 70, 80, 90]])
# -----------------------------------import numpy as np
#
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# #-----------------------
# # array([[1, 2, 3],
# #        [4, 5, 6],
# #        [7, 8, 9]])
# #-----------------------
#
# b = np.array([[10,20,30],[40,50,60],[70,80,90]])
# #-----------------------
# # array([[10, 20, 30],
# #        [40, 50, 60],
# #        [70, 80, 90]])
# #-----------------------
#
# np.vstack((a,b))
# #-----------------------
# # array([[ 1,  2,  3],
# #        [ 4,  5,  6],
# #        [ 7,  8,  9],
# #        [10, 20, 30],
# #        [40, 50, 60],
# #        [70, 80, 90]])
# #-----------------------
#
# np.hstack((a,b))
# #-----------------------------------
# # array([[ 1,  2,  3, 10, 20, 30],
# #        [ 4,  5,  6, 40, 50, 60],
# #        [ 7,  8,  9, 70, 80, 90]])
"""
