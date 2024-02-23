import matplotlib.pyplot as plt

# 数据
x = [10000,20000,40000,60000,80000,100000,120000]
y = [47344, 48090, 29608, 21820, 17487, 14268,11865]

# x = [64,128,256,512]
# y = [1805, 857, 330, 3400]
# 创建折线图，并更改线的粗细为2
plt.plot(x, y, linewidth=2, color= 'red', marker='x', markerfacecolor='yellow',markersize=9)

# 添加标签和标题
plt.xlabel('The value of Q',fontsize=16)
plt.ylabel('Judgement per minute(times/minute)',fontsize=16)
#plt.title('Line Plot with Increased Width')

# 显示图形
plt.show()