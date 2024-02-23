import matplotlib.pyplot as plt

# 数据
categories = ['LoRa', 'RivIs', 'Csei', 'RBS', 'Rand-conv', 'Mixtype', 'RFIS']
values = [278099, 413015, 99363, 271208, 262206, 107924, 51431]
colors = ['red', 'm', 'blue', 'green', 'orange','yellow', 'purple']  # 每个柱子的颜色

# 创建柱状图
plt.bar(categories, values, color=colors)

# 添加标签和标题
plt.xlabel('Method',fontsize=16)
plt.ylabel('Judgment in one minute(times/minute)',fontsize=16)
#plt.title('Speed of Method(Per second)',fontsize=20, fontweight='bold')

plt.xticks(fontsize=10)
# 显示图形
plt.show()