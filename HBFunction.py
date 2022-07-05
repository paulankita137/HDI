# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 09:23:56 2021

@author: ANKITA PAUL
"""





import numpy as np
import pandas as pd
#from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import append
import csv
import numpy as np
from numba import jit, cuda

#@jit(nopython=True)
#@cuda.jit

#@jit(forceobj=True)
@jit(target_backend = 'cuda',forceobj=True)

#@cuda.jit('void(int32[:]')
def hbfunction(dataset):
    
    #X=dataset.iloc[:,3]
    X=dataset
    #x= X[~np.isnan(X)]
    x=X
    x=(np.array(x)).astype(int)

    i=0
    j=1
    X[1]
    desat=[]
# for i in range(1,28919):
#     for j in (2,28920):
#         if X[j]<X[i]:
#             desat=desat.append(X[j])
#             j+=1
#             i+=1
#         else:
#             i+=1
#             j+=1
            
            
            
            
 
    

    arr = np.array([2, 0, 0, 0, 0, 1, 0, 1, 0, 0])

    seq = np.array([0,0])
    val1 = np.array([88,87,86,85,84,83,82,81,82,83,84,85,86,87,88])
    val2 = np.array([88,87,86,85,84,83,82,81,80,79,78,77,76,74])
    val1 = np.array([88,87,86,85,84,83,82,81])
    val1 = np.array([88,87,86,85,84,83,82,81])
    val1 = np.array([88,87,86,85,84,83,82,81])
    val=[97.26558328	,97.26558328,	96.09369039,	96.09369039,	96.09369039,	96.09369039	,95.31242847,	95.31242847,	95.31242847	,95.31242847	,95.31242847	,95.31242847,	95.31242847	,95.31242847,	96.09369039,	95.31242847,	96.09369039,	95.31242847,	95.31242847,	95.31242847,	95.31242847	,95.31242847	,94.14053559	,94.14053559	,93.35927367,	93.35927367,	93.35927367,	93.35927367	,92.18738079,	92.18738079	,92.18738079	,92.18738079,	91.01548791	,92.18738079,	92.18738079,	92.18738079	,92.18738079,	92.18738079,	92.18738079,	92.18738079	,92.18738079,	91.01548791	,91.01548791	,91.01548791,	90.23422599,	90.23422599,	90.23422599	,91.01548791	,91.01548791,	91.01548791,	91.01548791,	91.01548791	,91.01548791	,91.01548791,	91.01548791,	91.01548791,	91.01548791,	90.23422599,	90.23422599,	90.23422599,	90.23422599	,90.23422599,	90.23422599	,90.23422599	,91.01548791	,91.01548791	,91.01548791,	91.01548791	,91.01548791	,91.01548791	,91.01548791,	91.01548791	,91.01548791	,90.23422599,	90.23422599	,90.23422599,	90.23422599	,90.23422599	,90.23422599	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331,	89.0623331	,89.0623331	,89.0623331	,89.0623331	,88.28107118,	88.28107118,	88.28107118,	88.28107118	,88.28107118	,88.28107118	,88.28107118	,88.28107118	,88.28107118,	88.28107118	,89.0623331	,89.0623331	,89.0623331,	88.28107118,	88.28107118	,88.28107118	,88.28107118	,88.28107118,	88.28107118	,88.28107118	,88.28107118	,87.1091783	,87.1091783,	87.1091783	,87.1091783	,87.1091783	,87.1091783	,87.1091783	,86.32791638	,86.32791638	,86.32791638	,86.32791638,	85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,86.32791638	,86.32791638,	86.32791638	,86.32791638	,86.32791638	,86.32791638,	87.1091783	,87.1091783	,87.1091783	,88.28107118,	90.23422599	,91.01548791	,91.01548791	,93.35927367	,94.14053559,	95.31242847,	96.09369039,	96.09369039	,97.26558328	,97.26558328]
    val=(np.array(val)).astype(int)
#desat=[88,88,88,87,87,87,86,86,86,85,85,85,84,84,84,83,83,83,82,82,82,81,81,81,80,80,80,81,81,82,82,82,83,83,83,84,84,84,85,85,85,86,86,86,87,87,87,88,88,88]
#desat=[88,88,88,87,87,87,86,86,86,85,85,85,86,86,86,87,87,87,88,88,88]
    desat=[88,88,88,87,87,87,86,86,86,85,85,85,84,84,84,83,83,83,82,82,82,81,81,81,80,80,80,79,79,78,78,76,76,74,74,76,76,78,78,79,79,78,76,74,72,71,70,69,68,69,70,71,72,74,76,78,80,80,81,81,82,82,82,83,83,83,84,84,84,85,85,85,86,86,86,87,87,87,88,88,88]
#l=len(desat)
#seq = np.array([98,97,96,95,94,93,92,91,90,89,88,87,86,85,86,87,88,89,90,91,92,93,94,95,96])


#Function for calculating desaturation range
    eightyeight=0
    lengths=[]
    k = 39 #number of time spents we will be storing
    lists = [[] for _ in range(k)]#creates k number of lists

    for m in range(50,89):
        n=np.isin(x,m)
        print('Time spent at '+ str(m),(len(x[n])))
        

        lists[m-50].append(len(x[n]))
        
        lengths.append(len(x[n]))
        #np.savetxt("lengths.csv"+str(m), lengths, delimiter=",")
        eightyeight=eightyeight+len(x[n])
        m+=1

    times_list = lists
    print(times_list)
    SleepTime = len(x)/60

    love=np.array_split(x, 440)
    ratioarr=[]
    listdesat=[]

    for i in range (1,440):
    
        for j in range (0,len(love[i])):
        #print('j is',j)
            whoa=love[i]
            whoa1=whoa[j]//10
            if whoa1<=8:
               listdesat.append(love[i])
            else:
               j+=1
        i+=1



    h=len(listdesat)-1
    #print('length of listdesat before pop',len(listdesat))
#rint('listdesatbefore',listdesat)
    new_list=[]

    for ind in range ( 0,h):


        size = len(listdesat)
        array1= listdesat[ind]
   
   # print('array1',array1)
        l= len(array1)
   #print(l)
        ankita=0
        for i in range(1,l):
            ankita=0
            b= array1[i] // 10
            # print('b',b)
            if b==8:
               ankita=1
               new_list=np.append(new_list,[listdesat[ind]])
           
               break
            else :
                i+=1
            if ankita ==1:
     #listdesat.pop(ind)
               new_list=np.append(new_list,[listdesat[ind]])

       #print('after pop',ld)
       
    
        ind+=1
            
    

    time=np.array(range(1,(len(new_list)+1)))
#print(time)
# def fillArray(value, len) :
 
#   a = [value]
#   while ((len(a) * 2) <= len) :
#          a = a.concat(a)
#          if (a.length < len):
#             a = a.concat(a.slice(0, len - a.length));
#   return a;


# a=fillArray(100,1266)

#a = np.empty(1278)

    a = np.empty(len(time))

    a.fill(98)

    time=np.array(range(1,(len(new_list)+1)))
    #plt.ylim([60, 105])
#plt.xlim([-10, 2288])

   # plt.plot(time,a)
   # plt.legend(dataset)
    #area1=metrics.auc(time,a)
    #print('Baseline Area',area1)
    #print('Baseline Area Log Scaled',np.log(area1))


    #for n in range(1,len(time)):
     #   baseline=[]
      #  baseline=np.append(baseline,n)



    if all([ v == 0 for v in lengths ]) :
       print ('indeed they are')
       return  0,  0 , 0,0,0,SleepTime,0
    else:
     Area=[]

     time=np.array(range(1,(len(new_list)+1)))
     area=metrics.auc(time,(new_list))
     print('Desaturation Area',area)
     print('Desaturation Area Log scaled',np.log(area))


     Area=np.append(Area,area)
     #plt.ylim([60, 105])
     #plt.xlim([-10, 88576])

     #plt.plot(time,new_list)



# for j in range (1,16):
#     array= array(new_list[j])
#     time=np.array(range(1,(len(array)+1)))
#     area=metrics.auc(time,(new_list[j]))
#     Area=np.append(Area,area)
#     j+=1
    
#print('Area',Area)
     SUM=sum(Area)
     #print('Hypoxic Burden',(area/SleepTime))

     #print('Hypoxic Burden',np.log(area/SleepTime))
     #print(SleepTime)
     sleeptime = SleepTime    
     return  area,  np.log(area) , (area/SleepTime),np.log(area/SleepTime),eightyeight,sleeptime,times_list


