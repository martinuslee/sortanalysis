#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import random
import time
import copy
import sys; sys.setrecursionlimit(10000)

ipSize = [800 , 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800] #Data Size
Size = [100 , 200, 400, 800, 1600, 3200] #Data Size


# In[3]:


randnum =[]
iprandnum = []

for i in ipSize:
    iprandnum.append(random.sample(range(1,1000000), i))


# ## 셸 정렬(shell sort) 알고리즘의 특징
# + #### 장점
#     + 연속적이지 않은 부분 리스트에서 자료의 교환이 일어나면 더 큰 거리를 이동한다. 따라서 교환되는 요소들이 삽입 정렬보다는 최종 위치에 있을 가능성이 높아진다.
#     + 부분 리스트는 어느 정도 정렬이 된 상태이기 때문에 부분 리스트의 개수가 1이 되게 되면 셸 정렬은 기본적으로 삽입 정렬을 수행하는 것이지만 삽입 정렬보다 더욱 빠르게 수행된다.
#     + 알고리즘이 간단하여 프로그램으로 쉽게 구현할 수 있다.
# #### 셸 정렬(shell sort)의 시간복잡도
# 
# + 시간복잡도를 계산한다면
# 
#     - 평균: T(n) = O(n^1.5)
#     - 최악의 경우: T(n) = O(n^2)
# 

# In[4]:


def gapInsertionSort(arr, start, gap):
    for target in range(start+gap, len(arr), gap):
        val = arr[target]
        i = target
        while i > start:
            if arr[i-gap] > val:
                arr[i] = arr[i-gap]
            else:
                break
            i -= gap
        arr[i] = val

def shellSort(arr):
    gap = len(arr) // 2
    while gap > 0:
        for start in range(gap):
            gapInsertionSort(arr, start, gap)
        gap = gap // 2


# In[5]:


def gapInsertionSort(arr, start, gap):
    for target in range(start+gap, len(arr), gap):
        val = arr[target]
        i = target
        while i > start:
            if arr[i-gap] > val:
                arr[i] = arr[i-gap]
            else:
                break
            i -= gap
        arr[i] = val

def shellSort2(arr):
    gap = 1
    while gap < len(arr):
            gap = 3*gap +1
            
    while gap > 0:
        for start in range(gap):
            gapInsertionSort(arr, start, gap)
        gap = gap // 3


# In[5]:


def quickSort1(arr): #피벗 처음 값
    
    if len(arr) <= 1:
        return arr
   # pivot = arr[len(arr) // 2]
    pivot = arr[0]
    lesser_arr, equal_arr, greater_arr = [], [], []
    
    for num in arr:
        if num < pivot:
            lesser_arr.append(num) #작은 리스트로 보내기
        elif num > pivot:
            greater_arr.append(num) #큰 리스트로 보내기
        else:
            equal_arr.append(num)

	#작은 리스트와 큰 리스트끼리 다시 재귀하여 퀵정렬
    return quickSort1(lesser_arr) + equal_arr + quickSort1(greater_arr)

def quickSort2(arr): #피벗 중간 값
    
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    lesser_arr, equal_arr, greater_arr = [], [], []
    
    for num in arr:
        if num < pivot:
            lesser_arr.append(num) #작은 리스트로 보내기
        elif num > pivot:
            greater_arr.append(num) #큰 리스트로 보내기
        else:
            equal_arr.append(num)

	#작은 리스트와 큰 리스트끼리 다시 재귀하여 퀵정렬
    return quickSort2(lesser_arr) + equal_arr + quickSort2(greater_arr)

def quickSort3(arr): #피벗 마지막 값
    
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)-1]
    lesser_arr, equal_arr, greater_arr = [], [], []
    
    for num in arr:
        if num < pivot:
            lesser_arr.append(num) #작은 리스트로 보내기
        elif num > pivot:
            greater_arr.append(num) #큰 리스트로 보내기
        else:
            equal_arr.append(num)

	#작은 리스트와 큰 리스트끼리 다시 재귀하여 퀵정렬
    return quickSort3(lesser_arr) + equal_arr + quickSort3(greater_arr)

def quickSort4(arr): #피벗 랜덤 값
    
    if len(arr) <= 1:
        return arr
    pivot = arr[random.randint(0,len(arr)-1)]
    #### 랜덤 피벗 ####
    lesser_arr, equal_arr, greater_arr = [], [], []
    
    for num in arr:
        if num < pivot:
            lesser_arr.append(num) #작은 리스트로 보내기
        elif num > pivot:
            greater_arr.append(num) #큰 리스트로 보내기
        else:
            equal_arr.append(num)

	#작은 리스트와 큰 리스트끼리 다시 재귀하여 퀵정렬
    return quickSort4(lesser_arr) + equal_arr + quickSort4(greater_arr)

def quickSort5(arr): #피벗 처음 중간 마지막의 평균 값
    
    if len(arr) <= 1:
        return arr
    pivot = arr[(0+(len(arr)//2)+(len(arr)-1))//3]
    lesser_arr, equal_arr, greater_arr = [], [], []
    
    for num in arr:
        if num < pivot:
            lesser_arr.append(num) #작은 리스트로 보내기
        elif num > pivot:
            greater_arr.append(num) #큰 리스트로 보내기
        else:
            equal_arr.append(num)

	#작은 리스트와 큰 리스트끼리 다시 재귀하여 퀵정렬
    return quickSort5(lesser_arr) + equal_arr + quickSort5(greater_arr)


# In[6]:


cTime = []
cTime2 = []
randnum = []

for i in Size:
    randnum.append(random.sample(range(1,10000), i))
    
randnum_shell = copy.deepcopy(randnum)    
randnum_first = copy.deepcopy(randnum)
randnum_center = copy.deepcopy(randnum)
randnum_last = copy.deepcopy(randnum)
randnum_rand = copy.deepcopy(randnum)
randnum_median = copy.deepcopy(randnum)

for i in randnum:
    start = time.time()
    shellSort(i)
    end = (time.time() - start)*1000
    cTime.append(end)
    
for i in randnum_shell:
    start = time.time()
    shellSort2(i)
    end = (time.time() - start)*1000
    cTime2.append(end)

print(cTime, end='')
print('\n')
print(cTime2, end='')


# In[7]:


def drawGraph(x,y,z):
    plt.title("Random Data")
    plt.plot(x,y, label = "h = h / 2", marker = "*", markeredgecolor = 'red')
    plt.plot(x,z, label = "h = 3 * h + 1",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    
x = ["{}".format(i) 
     for i in Size]

drawGraph(x,cTime,cTime2)


# In[8]:


#이미 소팅된 데이터
qTime1 = []
qTime2 = []
qTime3 = []
qTime4 = []
qTime5 = []



for i in randnum_first:
    start = time.time()
    quickSort1(i) ## 처음값 피벗
    end = (time.time() - start)*1000
    qTime1.append(end)
for i in randnum_center:    
    start = time.time() 
    quickSort2(i) ## 중간값 피벗
    end = (time.time() - start)*1000
    qTime2.append(end)

for i in randnum_last:         
    start = time.time()
    quickSort3(i) ## 마지막값 피벗
    end = (time.time() - start)*1000
    qTime3.append(end)
for i in randnum_rand:
    start = time.time()
    quickSort4(i) ## 랜덤값 피벗
    end = (time.time() - start)*1000
    qTime4.append(end) 
for i in randnum_median:
    start = time.time()
    quickSort5(i)  ## Median 피벗
    end = (time.time() - start)*1000
    qTime5.append(end)

print('\n',qTime1,end ='') 
print('\n',qTime2,end ='') 
print('\n',qTime3,end ='') 
print('\n',qTime4,end ='') 
print('\n',qTime5,end ='') 


# In[9]:


def drawGraph(x,y1,y2,y3,y4,y5):
    plt.title("Quick Sort Pivot Comparision")
    plt.plot(x,y1, label = "Pivot Fisrt", marker = "*", markeredgecolor = 'red')
    plt.plot(x,y2, label = "Pivot Center",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y3, label = "Pivot Last",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y4, label = "Pivot Random",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y5, label = "Pivot Median",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    
x = ["{}".format(i) 
     for i in Size]

drawGraph(x,qTime1,qTime2,qTime3,qTime4,qTime5)


# In[10]:


def drawGraph(x,y,z):
    plt.title("Random Data")
    plt.plot(x,y, label = "Shell", marker = "*", markeredgecolor = 'red')
    plt.plot(x,z, label = "Quick",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    
x = ["{}".format(i) 
     for i in Size]

drawGraph(x,cTime2,qTime2)


# In[11]:


def drawGraph(x,y1,y2,y3,y4,y5):
    plt.title("Quick Sort Pivot Comparision")
    plt.plot(x,y1, label = "Pivot Fisrt", marker = "*", markeredgecolor = 'red')
    plt.plot(x,y2, label = "Pivot Center",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y3, label = "Pivot Last",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y4, label = "Pivot Random",marker = "*", markeredgecolor = 'red')
    plt.plot(x,y5, label = "Pivot Median",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.ylim(10,22)
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    
x = ["{}".format(i) 
     for i in Size]

drawGraph(x,qTime1,qTime2,qTime3,qTime4,qTime5)


# ## In-Place Sorting Method
#  

# In[22]:


def quick_sort(arr):
    def sort(low, high):
        if high <= low:
            return

        mid = partition(low, high)
        sort(low, mid - 1)
        sort(mid, high)

    def partition(low, high):
        pivot = arr[(low + high) // 2]

        while low <= high:
            while arr[low] < pivot:
                low += 1
            while arr[high] > pivot:
                high -= 1
            if low <= high:
                arr[low], arr[high] = arr[high], arr[low]
                low, high = low + 1, high - 1
        return low

    return sort(0, len(arr) - 1)


# In[13]:


ipqtime = []
ipqtime2 = []
for i in iprandnum:
    start = time.time()
    quick_sort(i)
    end = (time.time() - start)*1000
    ipqtime.append(end)
    
for i in randnum:
    start = time.time()
    quick_sort(i)
    end = (time.time() - start)*1000
    ipqtime2.append(end)
print(ipqtime, end = '')   #큰 데이터
print('\n',ipqtime2, end = '') #작은 데이터


# In[14]:


def drawGraph(x,y,z):
    plt.title("Sort Algorithms")
    plt.plot(x,y, label = "Shell", marker = "*", markeredgecolor = 'red')
    plt.plot(x,z, label = "in-Place Quick",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    
x = ["{}".format(i) 
     for i in Size]

drawGraph(x,cTime2,ipqtime2)


# In[18]:


s = [] # data size
ipqt =[] # in-place method quick soring 
shellt =[] # shell sorting

for i in range(0, 100100, 1000):
    s.append(i)

## 1,000 ~ 100,000, 1000씩 증가

qlist = list() # random sample
slist = list()

len(s)


# In[19]:


for i in range(0, len(s)):
    qlist.append(random.sample(range(1,100101),s[i]))
    
slist = copy.deepcopy(qlist)


# In[20]:


for i in qlist:
    start = time.time()
    quick_sort(i)
    end = (time.time()-start)*1000
    ipqt.append(end)

for j in slist:
    start = time.time()
    shellSort2(j)
    end = (time.time()-start)*1000
    shellt.append(end)


# In[18]:


plt.figure(1)
plt.title("<in- Place sorting method> Quick 0 ~ 100,000")
plt.plot(s,ipqt, label = "in-Place Quick",marker = "*", markeredgecolor = 'red')
plt.xlabel("Data Size")
plt.ylabel("Time(msecs)")
plt.xticks(ha = "center")
plt.grid(True, axis = "y")
plt.legend(loc = 'upper left')


plt.figure(2)
plt.title("Shell Sort 0 ~ 100,000")
plt.plot(s,shellt, label = "Shell Sort",marker = "*", markeredgecolor = 'red')
plt.xlabel("Data Size")
plt.ylabel("Time(msecs)")
plt.xticks(ha = "center")
plt.grid(True, axis = "y")
plt.legend(loc = 'upper left')
plt.show()


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

x1_train=np.array(s)
y1_train=np.array(ipqt)

line_fitter1 = LinearRegression()
line_fitter1.fit(x1_train.reshape(-1,1),ipqt)



plt.grid(True, axis = "y")
plt.xlabel("Data Size")
plt.ylabel("Time(msecs)")

plt.plot(x1_train,y1_train,'o')
line1 = plt.plot(x1_train,line_fitter1.predict(x1_train.reshape(-1,1)), label = 'quick')

x2_train=np.array(s)
y2_train=np.array(shellt)

line_fitter2 = LinearRegression()
line_fitter2.fit(x2_train.reshape(-1,1),shellt)

plt.plot(x2_train,y2_train,'o')
line2 = plt.plot(x2_train,line_fitter2.predict(x2_train.reshape(-1,1)), label = 'shell')
plt.legend()
plt.show()


# In[21]:


for i in qlist:
    start = time.time()
    shellSort(i)
    end = (time.time()-start)*1000
    ipqt.append(end)

for j in slist:
    start = time.time()
    shellSort2(j)
    end = (time.time()-start)*1000
    shellt.append(end)

def drawGraph(x,y,z):
    plt.title("Random Data")
    plt.plot(x,y, label = "Shell 2.1", marker = "*", markeredgecolor = 'red')
    plt.plot(x,z, label = "Shell 2.2",marker = "*", markeredgecolor = 'red')
    plt.xlabel("Data Size")
    plt.ylabel("Time(msecs)")
    plt.xticks(ha = "center")
    plt.grid(True, axis = "y")
    plt.legend(loc = 'upper left')
    plt.show
    

drawGraph(s,shellt,ipqt)





# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


x1_train=np.array(s)
y1_train=np.array(ipqt)

line_fitter1 = LinearRegression()
line_fitter1.fit(x1_train.reshape(-1,1),ipqt)



plt.grid(True, axis = "y")
plt.xlabel("Data Size")
plt.ylabel("Time(msecs)")

plt.plot(x1_train,y1_train,'o')
line1 = plt.plot(x1_train,line_fitter1.predict(x1_train.reshape(-1,1)), label = '2.1')

x2_train=np.array(s)
y2_train=np.array(shellt)

line_fitter2 = LinearRegression()
line_fitter2.fit(x2_train.reshape(-1,1),shellt)

plt.plot(x2_train,y2_train,'o')
line2 = plt.plot(x2_train,line_fitter2.predict(x2_train.reshape(-1,1)), label = '2.2')
plt.legend()
plt.show()

