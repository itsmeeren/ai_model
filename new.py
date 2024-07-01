#one hot encoding


#
# import numpy as np
# def onehot(num,num_class):
#     onehot=np.zeros(num_class)
#     onehot[num]=1
#     return onehot


# onehot=onehot(5,10)
# print(onehot)

# list1=[x for x in range(1,11)]
# print(list1[1:1:-1])


# li=[c for c in range(1,11)]
# def new(*hello):
#     # li=[]
#     for hell in hello:
#         # li.append(hell)
#         print(hell)
# new([x if x not in li else 0 for x in range(1,5)])


# import numpy as np
# # import pandas as pd
# import pandas as pd
#
# df=pd.DataFrame(np.random.randn(10,3),columns=['a','b','c'])
# print(df.head())
# x=df.iloc[0:2,0:3]# 2  is row and 3 is coulmn
# print(x)

# a=np.zeros(3)
# print(a)
import numpy as np
# l1 = ['hi', 'hehe', 'hurray']
# l2 = [0, 1]
#
# l3 = []
# for x in l2:
#     l3.append(l1[x])
#
# # print(l3)
# area_of_square = lambda *side: side * side
# print(area_of_square(5))  # Output: 25
# import numpy as np
#
# x=np.arange(1,10).reshape(3,3)
# # print(x)
#
# # print(x[:2,2:3])
#
# for i in range(5):
#     if(i==4):
#         break
#     else:
#         print(i)

# def sum(x=0,y=0):
#     return x + y
# x=sum((2+2))
# print(x)

# def array_sum(arr):
#     total = 0
#     for i in range( len(arr)):
#         total = total + arr[i]
#     return total
# ar1=[1,2,3,4,5]
# # print(array_sum(ar1))
# x=map(array_sum, ar1)
# print(x)

# hi=input("Enter name")
# print(f"hi\n{hi}")
# text=('imaobot')
#
# text=text.split()
# print(text)

import regex as re