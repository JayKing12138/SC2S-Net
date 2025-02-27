import numpy as np
import matplotlib.pyplot as plt

# 版本 & 组件
versions = ["(A)", "(B)", "(C)", "(D)"]
components = ["CNN", "SAM", "FTM", "FFM"]

# 组件存在性矩阵 (1 = 存在, 0 = 不存在)
data = np.array([
    [1, 0, 0, 0],  # A: 只有 CNN
    [1, 1, 0, 0],  # B: +SAM
    [1, 1, 1, 0],  # C: +FTM
    [1, 1, 1, 1]   # D: +FFM
])

# 颜色映射（自定义不同模块的颜色）
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 蓝, 橙, 绿, 红
labels = ["CNN Backbone", "SAM Encoder", "FTM", "FFM"]

# 创建堆叠柱状图
fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
bottom = np.zeros(len(versions))

for i in range(len(components)):
    ax.bar(versions, data[:, i], label=labels[i], color=colors[i], bottom=bottom, edgecolor='black', linewidth=0.8)
    bottom += data[:, i]

# 设置标签和标题
ax.set_ylabel("Component Existence")
ax.set_title("Model Versions and Network Components", fontsize=10, fontweight="bold")
ax.set_yticks([])
ax.legend(loc="upper left", fontsize=8, frameon=False)

# 去掉边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存和显示
plt.tight_layout()
plt.savefig("network_components.pdf", bbox_inches='tight', dpi=600)
plt.show()
