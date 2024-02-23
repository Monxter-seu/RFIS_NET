import matplotlib.pyplot as plt

# 数据
x = [64,128,256,512,1024,2048,4096]
y = [859, 857, 857, 842, 835,843,840]

# x = [64,128,256,512]
# y = [1805, 857, 330, 3400]
# 创建折线图，并更改线的粗细为2
plt.plot(x, y, linewidth=2, color= 'red', marker='x', markerfacecolor='yellow',markersize=9)

# 添加标签和标题
plt.xlabel('The value of Q',fontsize=16, fontweight='bold')
plt.ylabel('Judgement per second',fontsize=16, fontweight='bold')
#plt.title('Line Plot with Increased Width')

# 显示图形
plt.show()