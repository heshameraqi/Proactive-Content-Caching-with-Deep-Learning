
import matplotlib.pyplot as plt
 
# x axis values
x = [100,200,300,400,500]
# corresponding y axis values
#lru = [10.15, 19.32, 27.98, 34.64, 41.98]
fifo = [9.7, 18.04, 25.69, 31.65, 38.11]
gds = [10.15, 19.32, 27.98, 34.64, 41.98]
#gdsf = [13.11, 25.92, 37.01, 44.65, 52.62]
lfuda = [13.11, 25.92, 37.01, 44.65, 52.62]
filter_lru = [12.72, 23.05, 31.98, 38.2, 44.37]
#th_lru = [10.15,19.32,27.98,34.64,41.98]
#expProb_lru = [10.15,19.32,27.98,34.66,42]
lru_k = [15.74,28.97,39.29,46.59,53.86]
#adaptsize = [10.16,19.36,28.05,34.7,42.07]
foo = [38.27,51.57,60.11,66.45,71.45]
twostages = [22.15,35.27,45.41,53.22,59.81]
e2e = [22.62,36.33,46.39,54.43,61.19]
 
# plotting the points
plt.figure(figsize=(9, 7))
"""
plt.plot(x, lru, color='green', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='green', markersize=5, label = "LRU")
"""

plt.plot(x, fifo, color='green', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='green', markersize=5, label = "FIFO")

plt.plot(x, gds, color='yellow', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='yellow', markersize=5, label = "GDS")
"""
plt.plot(x, gdsf, color='magenta', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='magenta', markersize=5, label = "GDSF")
"""
plt.plot(x, lfuda, color='cyan', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='cyan', markersize=5, label = "LFUDA")

plt.plot(x, filter_lru, color='black', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='black', markersize=5, label = "Filter-LRU")
"""
plt.plot(x, th_lru, color='black', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='black', markersize=5, label = "TH-LRU")
"""
"""
plt.plot(x, expProb_lru, color='yellow', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='yellow', markersize=5, label = "ExpProb-LRU")
"""
plt.plot(x, lru_k, color='magenta', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='magenta', markersize=5, label = "LRU-K")
"""
plt.plot(x, adaptsize, color='cyan', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='cyan', markersize=5, label = "AsaptSize")
"""
plt.plot(x, foo, color='red', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='red', markersize=5, label = "FOO")

plt.plot(x, twostages, color='blue', linestyle='dashed', linewidth = 1,
         marker='o', markerfacecolor='blue', markersize=5, label = "DLC-TwoStage")

plt.plot(x, e2e, color='blue', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='blue', markersize=5, label = "DLC-E2E")

 
# setting x and y axis range
#plt.ylim(1,8)
#plt.xlim(1,8)
 
# naming the x axis
plt.xlabel('Cache Size', fontsize=14)
# naming the y axis
plt.ylabel('Hite Rate Ratio', fontsize=14)

# show a legend on the plot
plt.legend()
 
# giving a title to my graph
#plt.title('Some cool customizations!')
 
plt.savefig("CFR_benchmarking.png",bbox_inches='tight', dpi=300)