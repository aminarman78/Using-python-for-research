# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:29:18 2020

@author: Amin
"""
# def add(a,b):
#     mysum = a+b
#     return mysum
# add(2,3)
# print(add(5,6))

# def intersection(s1,s2):
#     lst = []
#     for x in s1:
#         if x in s2:
#             lst.append(x)
#     return lst
# print(intersection([1,2,3], [3,4,5,6,7]))

# s1 = [1,2,3,4,5,6,7]
# s2 = [5,6,7,8,9,10]
# print(intersection(s1,s2))
            

# =============================================================================
# 
# =============================================================================

# import random

# def password(length):
#     pw = str()
#     characters = 'abdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#     for i in range(length):
#         pw += random.choice(characters)
#     return pw

# print(password(10))
# =============================================================================
# 
# =============================================================================
    
# def is_vowel(letter):
#     if letter  in "aeiouy" :
#         return(True)
#     else:
#         return(False)

#   print(is_vowel('a'))
# =============================================================================
# 
# =============================================================================

    
# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         N = 1
#         for i in range(1,n+1):
#             N *= i
#         return N 

# print(factorial(4))
# =============================================================================
# 
# ============================================================================

# import string
# alphabet = string.ascii_letters 
# keys = []
# for letters in alphabet:
#     keys.append(letters)
# # print(keys)
# count_letters = {}


# def counter(input_string):
#     count_letters = {}
#     for letter in input_string:
#         if letter in alphabet:
#             if letter in count_letters:
#                 count_letters[letter] += 1
#             else:
#                 count_letters[letter] = 1
#     return count_letters

# cnt=counter("Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.")
# val = list(cnt.values())
# keysa = list(cnt.keys())
# maxval = max(val)
# print(keysa[val.index(maxval)])   
# =============================================================================
# 
# =============================================================================

# import math
# import random
# random.seed(1)

# def rand():
#     return(random.uniform(-1,1))
    

# def distance(a,b):
#     return ((a[0]-b[0])**2 + (a[1]-b[1])**2)*0.5

# def in_circle(x, origin = [0,0]):
#    if distance(x,origin) < 1:
#        return True
#    else:
#        return False

# totalnods= []

# for i in range(100):
#         totalnods.append((rand(),rand()))
# inside = []
# for nods in totalnods:
#     if in_circle(nods) == True:
#         inside.append(nods)

# print(totalnods)
# print('    \n\n\n\n\n\n')
# print(inside)
# print(len(inside)/len(totalnods))
       
# inside = []
# R = 10000
# for i in range(R):
#     point = [rand(), rand()]
#     inside.append(in_circle(point))

# print(sum(inside) / R)
# print(0.779-math.pi/4)
       
   
    
# def increment(n):
#     n += 1
#     return(n)

# n = 1
# while n < 10:
#     n = increment(n)
# print(n)
       
      
# class NewList(list):
#     def remove_max(self):
#         self.remove(max(self))
#     def append_sum(self):
#         self.append(sum(self))

# x = NewList([1,2,3])
# while max(x) < 10:
#     x.remove_max()
#     x.append_sum()

# print(x)


# import numpy
# x = numpy.array([[3,6],[5,7]])
# y = x.transpose()
# print(x)
# print(y)

# import numpy as np
# x = np.array([[1,2,3],[4,5,6]])
# print(x)
# print(x.size)
# print(x.shape)
# y = np.random.random(10)
# print(np.any(y < 0.9))
# print(np.all(y>0.1))
# print(y)
       
   
    
# =============================================================================
# 
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# plt.plot([0,1,4,9,16])
# plt.show()

# x = np.logspace(-1,1,40)
# y1 = x ** 2.0
# y2 = x ** 1.5
# plt.loglog(x, y1, 'bo-', linewidth=2, markersize=4, label='First')
# plt.loglog(x,y2,'gs-',linewidth=2,markersize=4, label='Second');
# # plt.axis([-5,15,-5,115])
# plt.xlabel('$X$')
# plt.ylabel('$Y$')
# plt.legend(loc='upper left')
# # plt.savefig('myplt.pdf')
# =============================================================================
# 
# =============================================================================
# import matplotlib.pyplot as plt
# import numpy as np

# # x = np.random.normal(size=10000)
# # print(plt.hist(x))
# # print(plt.hist(x, normed=True))
# # print(plt.hist(x,normed=True,bins = 23))
# # plt.grid(axis='x')
# x = np.random.gamma(2,3,100000)
# plt.figure()
# plt.subplot(221)
# plt.hist(x,bins = 30)
# plt.subplot(222)
# plt.hist(x, bins = 30, normed=True)
# plt.subplot(223)
# plt.hist(x, bins = 30,cumulative = True)
# plt.subplot(224)
# plt.hist(x, bins = 30, cumulative=True, normed=True , histtype='step')

# =============================================================================
# 
# =============================================================================

# import random
# x = [range(1,7),range(1,9),range(1,11)]
# print(random.choice(random.choice(x)))

# import numpy as np
# import random
# import matplotlib.pyplot as plt
# # # x = random.choice(range(1,7))
# # rolls = []
# # for k in range(100):
# #     rolls.append(random.choice(range(1,7)))
# # y = plt.hist(rolls, bins=np.linspace(0.5,6.5,7))

# ys = []
# for rep in range(10000):
#     y = 0
#     for k in range(10):
#         x=random.choice(range(1,7))
#         y += x
#     ys.append(y)

# plt.hist(ys)




# import numpy as np
# import matplotlib.pyplot as plt
# x = np.random.randint(1,7,(100,10))
# y = np.sum(x,axis=1)
# z = plt.hist(y)
# =============================================================================
# 
# =============================================================================

# import random
# import matplotlib.pyplot as plt
# import time
# start_time = time.clock()
# ys = []
# for rep in range(1000000):
#     y = 0
#     for k in range(10):
#         x=random.choice(range(1,7))
#         y += x
#     ys.append(y)

# plt.hist(ys)
# end_time = time.clock()
# print(end_time - start_time)

# start_time = time.clock()
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.random.randint(1,7,(1000000,10))
# y = np.sum(x,axis=1)
# z = plt.hist(y)
# end_time = time.clock()
# print(end_time - start_time)

# =============================================================================
# 
# =========================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# delta_x = np.random.normal(0, 1, (2, 1000000))
# x_0 = np.array([[0], [0]])
# x = np.concatenate((x_0, np.cumsum(delta_x, axis=1)), axis=1)
# plt.plot(x[0], x[1], 'ro-')
# plt.savefig('rw.pdf')

# import numpy as np
# import random
# import time
# import matplotlib.pyplot as plt

# def create_board():
#     return np.zeros((3,3), dtype=int)
    
# def place(board, player, position):
#     if board[position] == 0:
#         board[position] = player
        
# def possibilities(board):
#     not_occupied = []
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if board[i][j] == 0:
#                 not_occupied.append((i,j))
#     return not_occupied

# def random_place(board, player):
#     position = random.choice(possibilities(board))
#     place(board, player, position)
#     return board # not sure why is needed to return board

# def row_win(board, player):
#     map1 = map(lambda row: all(list(map(lambda playerNum: playerNum == player, row))), board)
#     return any(list(map1))

# def col_win(board, player):
#     return row_win(np.transpose(board), player)

# def diag_win(board, player):
#     diags = [[],[]]
#     n = len(board)
#     for i in range(n):
#         diags[0].append(board[i][i])
#         diags[1].append(board[i][n-i-1])
            
#     m = map(lambda row: all(list(map(lambda playerNum: playerNum == player, row))), diags)
#     return any(list(m))

# def evaluate(board):
#     result = 0
    
#     for player in [1,2]:
#         if row_win(board, player) or col_win(board, player) or diag_win(board, player):
#             return player
    
#     if len(possibilities(board)) == 0:
#         result = -1
        
#     return result

# def play_game(strat):
#     board = create_board()
#     result = 0
#     counter = 0
    
#     if strat:
#         place(board, 1, (1,1))
#         counter += 1
        
#     while result == 0:
#         random_place(board, counter % 2 + 1)
#         result = evaluate(board)
#         counter += 1

#     return result

# # start = time.time()
# # results = [[],[]]
# # for i in range(10000):
# #     results[0].append(play_game(False))
# #     results[1].append(play_game(True))

# # print(results)
# # plt.hist(results)
# # stop = time.time()
# # print(stop - start)
# random.seed(1)
# results = []
# for i in range(1000):
#    results.append(play_game(0))
    
# count = 0
# for i in results:
#     if i==1:
#         count += 1
# print(count)




# =============================================================================
# 
# =============================================================================
