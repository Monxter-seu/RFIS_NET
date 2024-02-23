import matplotlib.pyplot as plt

# 数据
categories = ['LoRa', 'RivIs', 'Csei', 'RBS', 'Rand-conv', 'Mixtype', 'RFIS']
values = [8.8, 8.5, 9.1, 8.9, 8.9, 8.0, 9.2]
colors = ['red', 'm', 'blue', 'green', 'orange','yellow', 'purple']  # 每个柱子的颜色

# 创建柱状图
plt.bar(categories, values, color=colors)

# 添加标签和标题
plt.xlabel('Method',fontsize=16)
plt.ylabel('CPU Usage(%)',fontsize=16)
#plt.title('Speed of Method(Per second)',fontsize=20, fontweight='bold')

plt.xticks(fontsize=10)
# 显示图形
plt.show()