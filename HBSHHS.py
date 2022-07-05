# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 00:44:07 2021

@author: ANKITA PAUL
"""




import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy import append
import HBFunction 
import csv

#names1=[91]
#names2=[423,	440,	441,	449,	452,	453,	466,	469,	475,	476,	509,	558,	561,	582]
#names1=[26,27,28,29,31,32,34,36,37,38,42,43,44,45,46,47,48,49,50]
#names1=[441,	449,	452,	453,	466,	469,	475,	476,	509,	558,	561,	582,	606,	611,	613,	614,	655,	662,	692,	706,	710,	727,	731,	736,	744,	758,	770,	789,	801,	817,	858,	878,	928,	938,	941,	962]
#names1=[71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,88,89,90]
#names1=[412,	413,	414,	416,	417,	418,	419,	420,	421,	422,	424,	426,	427,	428,	430,	431,	433,	434,	435,436,			439,	442,	443,	444,	445,	446,	447,	448,	450,	454,	455,	456,	457,	459,	460,	461,	462,	464,	465,	467,	468,	470,	472,	474,	477,	478,	479,	480,	481,	482,	483,	484,	485,	487,	490,	491,	493,	494,	495,	496,	497,	498,	499,		501,	502,	504,	505,	506,	507,	508,	510]
#names1=[606,	611,	613,	614,	655,	662,	692,	706,	710,	727,	731,	736,	744,	758,	770,	789,	801,	817,	858,	878,	928,	938,	941,	962]
#names1=[1025,	1048,	1072,	1087,	1113,	1118,	1126,	1137,	1191,	1195,	1196,	1198,	1201,	1202,	1206,	1214,	1222,	1225]
#names1=[1230,	1233,	1237,	1238,	1241,	1243,	1245,	1248,	1252,	1254,	1255,	1256,	1266,	1270,	1276,	1281,	1283,	1295,	1299,	1300,	1304,	1305,	1306,	1313,	1316,	1318,	1319,	1324,	1329,	1337,	1338,	1342,	1354,	1357,	1359,	1360,	1361,	1365,	1368,	1375]
#names1=[1378,	1381,	1382,	1385,	1387,	1393,	1405,	1419,	1425,	1431,	1432,	1437,	1439,	1447,	1452,	1458,	1473,	1478,	1483,	1492,	1496,	1499,	1503,	1506,	1507,	1509,	1511,	1515,	1519,	1520,	1523,	1527,	1530,	1531,	1532,	1533,	1538,	1539,	1542,	1544,	1546,	1549,	1551,	1552,	1554,	1555,	1559,	1560,	1599,	1602,	1604,	1607,	1609,	1611,	1618,	1621,	1632,	1639,	1641,	1644,	1652,	1653,	1656,	1658,	1660,	1662,	1663,	1664,	1670,	1671,	1672,	1674,	1680,	1685,	1687,	1690,	1691,	1692,	1694,	1696,	1697]
#names1=[1705,	1707,	1709,	1713,	1716,	1791,	1794,	1830,	1833,	1834,	1837,	1843,	1849,	1853,	1854,	1855,	1857,	1858,	1864,	1866,	1867,	1868,	1870,	1871,	1873,	1874,	1879,	1881,	1895,	1897,	1898	]
#names1=[2116,	2119,	2121,	2123,	2147,	2151,	2154,	2156,	2158,	2160,	2167,	2174,	2179,	2183,	2186,	2187,	2196,	2205,	2208]
#names1=[	110,	116,	124,	139,	140,	180,	192,	193,	201,	208,	209,	228,	297,	298,	312,	377,	392,	423,	440]
#names1=[77,	78,	79,	80,	81,	82,	83,	84,	85,	86,	88,	89,	90,	92,	93,	95,	96,		98,	99]
#names1=[	100,	101,	102,	103,	105,	106,	107,	108,	111,	112,	113,	114,	115,	117,	121,	122,	123,	125,	126,	127,	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	141,	142,	144,	145,	147,	148,	149,	150,	151,	152,153,	154,	155,	156,	157,	158,	159,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170]
#names1=[171,	173,	174,	175,	176,	177,	178,	179,	181,	183,	185,	186,	187,	189,	191,	194,	195,	196,	197,	198,	199,	200,	202,	207,	210,	211,	212,	213,	214,	215,	216,	217,	218,	219,	220,	221,	223,	224,	225,	226,	227,	229,	231,	232,	233,	242,	243,	291,	292,	293,	294,	295,	296,	299,	300,	302,	303,	304,	305,	306,	307,	308,	309,	311,	314,	315,	316,	317,	318,	319,	320,	321,	322,	323,	324,	325,	326,	327,	328,329,333,	336,	337,	338,	339,	340,	341,	342,	343,	345,	346,	347,	349,	350,	352,	353,	354,	355,	356,	358,	359,	360,	361,	362,	363,	364,	365,	373,	374,	375,	376,	378,	380,	381,	383,	385,	386,		388,	389,	390,	393,	394,	395,	396,	397,	398,	401,	403,	404,	405,	407,	408,	409,	410]
names1=[600,601,602,603,604,605,606,607,608,609]
#names1=[104,	109,	184,	190,	310,	313,	351,	382,	406,	411,	463,	471,	492,	515,	652,	693,	705,	708,	709,	757,	786,	813,	830,	863,	875,	907,	948,	957,	977,	991,	]
#names1=[	513	,516	,517	,518	,519	,520	,521	,522	,523	,524	,555	,556	,557	,559	,560	,562	,563	,564	,566	,568	,569	,570	,571	,572	,573	,575	,576	,577	,578	,579	,580	,581	,583	,584	,585	,586	,587	,588	,589,590	,	593	,594	,595	,596	,597	,598	,599	,601	,602	,603	,604	,605	,607	,608	,609	,610	,612	,615]
#names1=[	813,	830,	863,	875,	907,	948,	957,	977,	991	]
#names1=[110,	116,	124,	139,	140,	180,	192,	193,	201,	208,	209,	228,	297,	298,	312,	377,	392]
#names1=[1001,	1032,	1098,	1109,	1134,	1190,	1193,	1199,	1204,	1207,	1210,	1211,	1218,	1219,	1223,	1224,	1229,	1240,	1242,	1244,	1249]
#names2=[110,	116,	124,	139,	140,	180,	192,	193,	201,	208,	209,	228,	297,	298,	312,	377,	392]
#names2=[1252	,1253	,1254	,1255	,1256	,1260	,1261	,1262	,1266	,1269	,1270	,1271	,1272	,1273	,1276	,1279	,1280	,1281	,1283	,1286	,1290	,1291	,1292	,1294	,1295	,1298	,1299	,1300	,1301	,1304	,1305	,1306	,1312	,1313	,1315	,1316	,1318	,1319	,1320	,1322	,1324	,1329	,1335	,1336	,1337	,1338	,1341	,1342	,1344	,1345	,1347	,1350]
#names2=[1354	,1356	,1357	,1359	,1360	,1361	,1362	,1365	,1366	,1368	,1369	,1374	,1375	,1376	,1377	,1378	,1379	,1381	,1382	,1384	,1385	,1387	,1393	,1396	,1397	,1399	,1400	,1401	,1402	,1403	,1404	,1405	,1409	,1411		,1415	,1417	,1419	,1421	,1423	,1424	,1425	,1428	,1431	,1432	,1433]
#names2=[1436	,1437	,1438	,1439	,1442	,1447	,1449	,1451	,1452	,1458	,1463	,1466	,1473	,1475,	1478	,1479	,1480	,1483		,1491	,1492	,1495	,1496	,1497	,1499	,1502	,1503	,1506	,1507	,1509	,1511	,1514	,1515	,1518	,1519	,1520	,1523	,1527	,1528	,1530	,1531	,1532	,1533	,1534	,1535	,1536	,1538	,1539	,1541	,1542	,1544	,1546	,1549	,1551	,1552	,1554	,1555	,1558	,1559	,1560]
#names2=[1599	,1600	,1602	,1604	,1607	,1609	,1611	,1618	,1621	,1622	,1626	,1627	,1632,	1633	,1637	,1639	,1641	,1644	,1646	,1647	,1648	,1650	,1652	,1653	,1656	,1658	,1660	,1662	,1663	,1664		,1670	,1671	,1672	,1674	,1679	,1680	,1685	,1687	,1690	,1691	,1692,	1694	,1696	,1697		,1705	,1706	,1707, 1708]
#names2=[1709	,1710	,1713	,1716	,1785	,1787	,1788	,1789	,1791	,1794	,1795	,1830	,1833	,1834	,1836	,1837	,1838	,1841	,1843	,1847	,1848	,1849	,1853	,1854	,1855	,1857	,1858	,1863	,1864	,1866	,1867	,1868	,1870	,1871	,1872	,1873	,1874	,1877	,1879	,1881	,1885	,1886	,1890	,1891	,1893	,1895	,1897	,1898	,1902]
#names2=[2108	,2109	,2110	,2116	,2117	,2118	,2119	,2121	,2122	,2123	,2128	,2134	,2141	,2147	,2148	,2151	,2154	,2156	,2157	,2158	,2160	,2161	,2162	,2166	,2167	,2168	,2170	,2171	,2174	,2176	,2179	,2183	,2186	,2187	,2189	,2190	,2191	,2196	,2199	,2201	,2202	,2205	,2206	,2208]
# desaturation_area=[]
# desaturation_area_lg=[]
# hypoxic_burden=[]
# hypoxic_burden_lg=[]
sleeptime=[]
Desat_Area=[] 
Hyp_Burden=[]
SleepTime=[]
EightyEight=[]
Baseline=[]
#empty lists to keep the csv columns
c1 = []
c2 = []
to_delete=[]
#dataset=pd.read_csv('shhs1-200091.edfSaO2_1sps.csv')

def newlist(reader):
    to_delete = [0]
    print(c2[0])
             #print('c20',c2[0])

    new_list = [float(val) for n, val in enumerate(c2) if n not in to_delete]
             #print('newlist0',new_list[0])
         #new_list=list(map(float, new_list))
         #new_list_=list(map(float, new_list))
    
        

    new_list = [ elem for elem in new_list if elem > 50 ]
    return new_list
    
for i in range (0,len(names1)):
    print('running'+str(i)+'th simulation')
    df=pd.read_csv('shhs1-200'+str(names1[i])+'.edfSaO2_1sps.csv',sep=';')
    X=df.iloc[:,1]
    new_list_ = [ elem for elem in X if elem > 50 ]

             
    
   
    dataset=new_list_
    
    desaturation_area,desaturation_area_lg,hypoxic_burden,hypoxic_burden_lg,baseline,eightyeight=HBFunction.hbfunction(dataset)
    Desat_Area.append(desaturation_area_lg)
    np.savetxt("desat.csv", Desat_Area, delimiter=",")
    Hyp_Burden.append(hypoxic_burden_lg)
    np.savetxt("Hyp.csv", Hyp_Burden, delimiter=",")
    Baseline.append(baseline)
    np.savetxt("Baseline.csv", Baseline, delimiter=",")
    EightyEight.append(eightyeight)
    np.savetxt("Eightyeight.csv", EightyEight, delimiter=",")


print(Desat_Area)
print(Hyp_Burden)
print(Baseline)
    

# X=dataset.iloc[:,3]
# x= X[~np.isnan(X)]
# x=(np.array(x)).astype(int)

# i=0
# j=1
# X[1]
# desat=[]
# # for i in range(1,28919):
# #     for j in (2,28920):
# #         if X[j]<X[i]:
# #             desat=desat.append(X[j])
# #             j+=1
# #             i+=1
# #         else:
# #             i+=1
# #             j+=1
            
            
            
            
            
# def search_sequence_numpy(arr,seq):
#     """ Find sequence in an array using NumPy only.

#     Parameters
#     ----------    
#     arr    : input 1D array
#     seq    : input 1D array

#     Output
#     ------    
#     Output : 1D Array of indices in the input array that satisfy the 
#     matching of input sequence in the input array.
#     In case of no match, an empty list is returned.
#     """

#     # Store sizes of input array and sequence
#     Na, Nseq = arr.size, seq.size

#     # Range of sequence
#     r_seq = np.arange(Nseq)

#     # Create a 2D array of sliding indices across the entire length of input array.
#     # Match up with the input sequence & get the matching starting indices.
#     M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

#     # Get the range of those indices as final output
#     if M.any() >0:
#         return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
#     else:
#         return []         # No match found
    
    
    

# arr = np.array([2, 0, 0, 0, 0, 1, 0, 1, 0, 0])

# seq = np.array([0,0])
# val1 = np.array([88,87,86,85,84,83,82,81,82,83,84,85,86,87,88])
# val2 = np.array([88,87,86,85,84,83,82,81,80,79,78,77,76,74])
# val1 = np.array([88,87,86,85,84,83,82,81])
# val1 = np.array([88,87,86,85,84,83,82,81])
# val1 = np.array([88,87,86,85,84,83,82,81])
# val=[97.26558328	,97.26558328,	96.09369039,	96.09369039,	96.09369039,	96.09369039	,95.31242847,	95.31242847,	95.31242847	,95.31242847	,95.31242847	,95.31242847,	95.31242847	,95.31242847,	96.09369039,	95.31242847,	96.09369039,	95.31242847,	95.31242847,	95.31242847,	95.31242847	,95.31242847	,94.14053559	,94.14053559	,93.35927367,	93.35927367,	93.35927367,	93.35927367	,92.18738079,	92.18738079	,92.18738079	,92.18738079,	91.01548791	,92.18738079,	92.18738079,	92.18738079	,92.18738079,	92.18738079,	92.18738079,	92.18738079	,92.18738079,	91.01548791	,91.01548791	,91.01548791,	90.23422599,	90.23422599,	90.23422599	,91.01548791	,91.01548791,	91.01548791,	91.01548791,	91.01548791	,91.01548791	,91.01548791,	91.01548791,	91.01548791,	91.01548791,	90.23422599,	90.23422599,	90.23422599,	90.23422599	,90.23422599,	90.23422599	,90.23422599	,91.01548791	,91.01548791	,91.01548791,	91.01548791	,91.01548791	,91.01548791	,91.01548791,	91.01548791	,91.01548791	,90.23422599,	90.23422599	,90.23422599,	90.23422599	,90.23422599	,90.23422599	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331	,89.0623331,	89.0623331	,89.0623331	,89.0623331	,89.0623331	,88.28107118,	88.28107118,	88.28107118,	88.28107118	,88.28107118	,88.28107118	,88.28107118	,88.28107118	,88.28107118,	88.28107118	,89.0623331	,89.0623331	,89.0623331,	88.28107118,	88.28107118	,88.28107118	,88.28107118	,88.28107118,	88.28107118	,88.28107118	,88.28107118	,87.1091783	,87.1091783,	87.1091783	,87.1091783	,87.1091783	,87.1091783	,87.1091783	,86.32791638	,86.32791638	,86.32791638	,86.32791638,	85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,85.1560235	,86.32791638	,86.32791638,	86.32791638	,86.32791638	,86.32791638	,86.32791638,	87.1091783	,87.1091783	,87.1091783	,88.28107118,	90.23422599	,91.01548791	,91.01548791	,93.35927367	,94.14053559,	95.31242847,	96.09369039,	96.09369039	,97.26558328	,97.26558328]
# val=(np.array(val)).astype(int)
# #desat=[88,88,88,87,87,87,86,86,86,85,85,85,84,84,84,83,83,83,82,82,82,81,81,81,80,80,80,81,81,82,82,82,83,83,83,84,84,84,85,85,85,86,86,86,87,87,87,88,88,88]
# #desat=[88,88,88,87,87,87,86,86,86,85,85,85,86,86,86,87,87,87,88,88,88]
# desat=[88,88,88,87,87,87,86,86,86,85,85,85,84,84,84,83,83,83,82,82,82,81,81,81,80,80,80,79,79,78,78,76,76,74,74,76,76,78,78,79,79,78,76,74,72,71,70,69,68,69,70,71,72,74,76,78,80,80,81,81,82,82,82,83,83,83,84,84,84,85,85,85,86,86,86,87,87,87,88,88,88]
# #l=len(desat)
# #seq = np.array([98,97,96,95,94,93,92,91,90,89,88,87,86,85,86,87,88,89,90,91,92,93,94,95,96])


# #Function for calculating desaturation range
# for m in range(70,89):
#     n=np.isin(x,m)
#     print('Time spent at '+ str(m),(len(x[n])))
#     m+=1

# # mask1 = np.isin(x, val2)
# # print('Time spent below 88',(len(x[mask1])/60))
# # mask11 = np.isin(x, 76)
# # print('Time spent in 81',len(x[mask11]))
# # mask10 = np.isin(x, 77)
# # print('Time spent in 77',len(x[mask10]))
# # mask9 = np.isin(x, 78)
# # print('Time spent in 78',len(x[mask9]))
# # mask8 = np.isin(x, 79)
# # print('Time spent in 79',len(x[mask8]))
# # mask7 = np.isin(x, 80)
# # print('Time spent in 80',len(x[mask7]))
# # mask2 = np.isin(x, 81)
# # print('Time spent in 81',len(x[mask2]))
# # mask3 = np.isin(x, 82)
# # print('Time spent in 82',len(x[mask3]))
# # mask4 = np.isin(x, 83)
# # print('Time spent in 83',len(x[mask4]))
# # mask5 = np.isin(x, 84)
# # print('Time spent in 84',len(x[mask5]))
# # mask6 = np.isin(x, 85)
# # print('Time spent in 85',len(x[mask6]))
# # mask7 = np.isin(x, 86)
# # print('Time spent in 86',len(x[mask7]))
# # mask8 = np.isin(x, 87)
# # print('Time spent in 87',len(x[mask8]))
# #print(mask)
# SleepTime = len(x)/60

# love=np.array_split(x, 440)

# ratioarr=[]
# listdesat=[]

# for i in range (1,440):
    
# #     ratio=fuzz.ratio(love[i], desat) 
# #     ratioarr=np.append(ratioarr,[ratio])
# #     #y=search_sequence_numpy(love[i],desat)
# #     #yarr=np.append(yarr,y)
# #     if ratio>40:
# #         listdesat.append(love[i])
# #     else :
# #         i+=1
        
# #     i+=1
    
    
#     for j in range (0,len(love[i])):
#         #print('j is',j)
#         whoa=love[i]
#         whoa1=whoa[j]//10
#         if whoa1<=8:
#             listdesat.append(love[i])
#         else:
#             j+=1
#     i+=1
# #print('Ratios',ratioarr)

# # y=np.where(x=val)
# #print('listdesatbefore',listdesat)

        
# # print('listdesatt',listd)

# #istdesat=[[99,98,99,98,97,96],[99,88,87,91],[99,98,88,97,95],[99,97,96,95]]


# h=len(listdesat)-1
# print('length of listdesat before pop',len(listdesat))
# #rint('listdesatbefore',listdesat)
# new_list=[]

# for ind in range ( 0,h):
#    #array1=[]
#    # print('ind',ind)

#     size = len(listdesat)
#     array1= listdesat[ind]
   
#    # print('array1',array1)
#     l= len(array1)
#    #print(l)
#     ankita=0
#     for i in range(1,l):
#         ankita=0
#         b= array1[i] // 10
#       # print('b',b)
#         if b==8:
#            ankita=1
#            new_list=np.append(new_list,[listdesat[ind]])
           
#            break
#         else :
#             i+=1
#     if ankita ==1:
#      #listdesat.pop(ind)
#       new_list=np.append(new_list,[listdesat[ind]])

#        #print('after pop',ld)
       
    
#     ind+=1
            
    
# #print('listdesatafterpop',len(new_list))
# # print('length of listdesat after',len(listdesat))

       
       
# # del listdesat[1:9]


# # print('listdesatsliced',listdesat)

# # del listdesat[18:22]

# # print('listdesatsecondsliced',listdesat)
# time=np.array(range(1,(len(new_list)+1)))
# #print(time)
# # def fillArray(value, len) :
 
# #   a = [value]
# #   while ((len(a) * 2) <= len) :
# #          a = a.concat(a)
# #          if (a.length < len):
# #             a = a.concat(a.slice(0, len - a.length));
# #   return a;


# # a=fillArray(100,1266)

# #a = np.empty(1278)

# a = np.empty(len(time))

# a.fill(100)
# #print(a)

# time=np.array(range(1,(len(new_list)+1)))
# plt.ylim([60, 105])
# #plt.xlim([-10, 2288])

# plt.plot(time,a)
# area1=metrics.auc(time,a)
# print('Baseline Area',area1)
# print('Baseline Area Log Scaled',np.log(area1))


# for n in range(1,len(time)):
#     baseline=[]
#     baseline=np.append(baseline,n)




# Area=[]

# time=np.array(range(1,(len(new_list)+1)))
# area=metrics.auc(time,(new_list))
# print('Desaturation Area',area)
# print('Desaturation Area Log scaled',np.log(area))


# Area=np.append(Area,area)
# plt.ylim([60, 105])
# plt.xlim([-10, 88576])

# plt.plot(time,new_list)



# # for j in range (1,16):
# #     array= array(new_list[j])
# #     time=np.array(range(1,(len(array)+1)))
# #     area=metrics.auc(time,(new_list[j]))
# #     Area=np.append(Area,area)
# #     j+=1
    
# #print('Area',Area)
# SUM=sum(Area)
# print('Hypoxic Burden',(area/SleepTime))

# print('Hypoxic Burden',np.log(area/SleepTime))
# print(SleepTime)

