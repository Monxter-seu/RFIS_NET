import matplotlib.pyplot as plt

# 数据
categories = ['LoRa', 'Csei', 'RBS', 'Rand-conv', 'Mixtype', 'RFIS']
values = [35, 41, 23, 42, 26, 39]
colors = ['red', 'blue', 'green', 'orange','yellow', 'purple']  # 每个柱子的颜色

# 创建柱状图
plt.bar(categories, values, color=colors)

# 添加标签和标题
plt.xlabel('Method',fontsize=16)
plt.ylabel('GPU Usage(%)',fontsize=16)
#plt.title('Speed of Method(Per second)',fontsize=20, fontweight='bold')

plt.xticks(fontsize=10)
# 显示图形
plt.show()