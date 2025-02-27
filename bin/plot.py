import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm

# 设置Nature期刊样式


# 指定 Arial 字体路径
# font_path = "/data/userdisk1/crq/font/Arial.ttf"
# font_prop = fm.FontProperties(fname=font_path)

rcParams.update({
    'font.size': 8,
    # 'font.sans-serif': font_prop.get_name(), 
    'font.sans-serif': 'Arial',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.spines.right': False,
    'axes.spines.top': False
})




# font_path = "/data/userdisk1/crq/font/Arial.ttf"
# font_prop = fm.FontProperties(fname=font_path)

# rcParams["font.sans-serif"] = font_prop.get_name()


# # 示例数据（需要替换为真实数据）
# methods = ['RecoBundles', 'Atlas', 'Atlas MRtrix','TractSeg', 'TransUNet', 'Swin-Unet','UCtransNet', 'SCC-Net']
# metrics = ['Dice Score', '95% HD (mm)']

# # 性能数据
# performance = {
#     'Dice Score': [0.6944, 0.6657, 0.7314, 0.8385, 0.8527, 0.8478, 0.8577, 0.8767],
#     '95% HD (mm)': [18.868, 14.457, 12.276, 9.216, 8.264,8.763,8.698,7.156, ]
# }

# # P值矩阵（6个方法 vs SAMTractNet）
# p_matrix = np.array([
#     [0.03, 0.04],  # RecoBundles
#     [0.02, 0.03],  # Atlas
#     [0.04, 0.02],  # Atlas_MRtrix
#     [0.01, 0.01],  # TractSeg
#     [0.008, 0.006],  # TransUNet
#     [0.005, 0.002],  # Swin-Unet
#     [0.006, 0.002] #uctransnet
# ])

# # 创建画布
# fig = plt.figure(figsize=(6, 6), dpi=300)
# gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.8)  # 三行布局

# # 颜色设置（蓝色系渐变）
# colors = plt.cm.Blues(np.linspace(0.4, 1, len(methods)))

# # ========== 第一行：Dice Score 柱状图 ==========
# ax_dice = plt.subplot(gs[0])
# x = np.arange(len(methods))  # 方法位置
# width = 0.8  # 更粗的柱宽

# # 绘制Dice Score柱状图
# bars_dice = ax_dice.bar(x, performance['Dice Score'], width,
#                         color=colors, edgecolor='black', linewidth=0.5)

# # 样式设置
# ax_dice.set_ylabel('Dice Score', labelpad=2)
# ax_dice.set_ylim(0.5, 0.95)
# ax_dice.set_xticks(x)
# ax_dice.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
# ax_dice.set_title('A) Dice Score', loc='left', pad=5, fontsize=9, fontweight='bold')

# # 添加数值标签
# for bar in bars_dice:
#     height = bar.get_height()
#     ax_dice.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
#                 f'{height:.3f}', ha='center', va='bottom', fontsize=6)

# # ========== 第二行：95% HD 柱状图 ==========
# ax_hd = plt.subplot(gs[1])

# # 绘制95% HD柱状图
# bars_hd = ax_hd.bar(x, performance['95% HD (mm)'], width,
#                     color=colors, edgecolor='black', linewidth=0.5)

# # 样式设置
# ax_hd.set_ylabel('95% HD (mm)', labelpad=2)
# ax_hd.set_ylim(0,20)
# ax_hd.set_xticks(x)
# ax_hd.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
# ax_hd.set_title('B) 95% Hausdorff Distance', loc='left', pad=5, fontsize=9, fontweight='bold')

# # 添加数值标签
# for bar in bars_hd:
#     height = bar.get_height()
#     ax_hd.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
#               f'{height:.3f}', ha='center', va='bottom', fontsize=6)

# # ========== 第三行：p值热力图 ==========
# ax_bottom = plt.subplot(gs[2])
# methods_short = methods[:-1]  # 排除SAMTractNet

# # 创建热力图数据
# heatmap_data = pd.DataFrame(p_matrix,
#                            index=methods_short,
#                            columns=metrics)

# # 绘制热力图
# sns.heatmap(heatmap_data, ax=ax_bottom, cmap='Blues_r',
#            annot=True, fmt=".3f", linewidths=0.5,
#            cbar_kws={'label': 'p-value', 'shrink': 0.8},
#            annot_kws={'size': 7, 'color':'black'})

# # 热力图样式调整
# ax_bottom.set_xticklabels(ax_bottom.get_xticklabels(), rotation=45,
#                         ha='right', rotation_mode='anchor')
# ax_bottom.set_yticklabels(ax_bottom.get_yticklabels(),
#                         rotation=0, va='center')
# ax_bottom.set_xlabel('Metrics', labelpad=5)
# ax_bottom.set_ylabel('Comparison Methods', labelpad=5)
# ax_bottom.set_title('C) Statistical Significance', loc='left', pad=5, fontsize=9, fontweight='bold')

# # 添加显著性标记
# for text in ax_bottom.texts:
#     p = float(text.get_text())
#     if p < 0.001:
#         text.set_text('<0.001***')
#     elif p < 0.01:
#         text.set_text(f'{p:.3f}**')
#     elif p < 0.05:
#         text.set_text(f'{p:.3f}*')
#     else:
#         text.set_text(f'{p:.3f}')

# # ========== 全局调整 ==========
# plt.tight_layout()
# plt.savefig('/home/crq/TractSeg/nature_style_split_plots.pdf', bbox_inches='tight')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import seaborn as sns
import pandas as pd

# ========== 设置 Nature 期刊风格 ==========
rcParams.update({
    'font.size': 8,
    'font.sans-serif': 'Arial',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.spines.right': False,
    'axes.spines.top': False
})

# ========== 示例数据 ==========
methods = ['RecoBundles', 'Atlas', 'Atlas MRtrix', 'TractSeg',  'Swin-Unet','TransUNet', 'UCtransNet', r'SC$^{2}$S-Net']
metrics = ['Dice Score', '95% HD (mm)']

# 方法性能数据
performance = {
    'Dice Score': [0.6944, 0.6657, 0.7314, 0.8385,  0.8478,0.8527, 0.8577, 0.8767],
    '95% HD (mm)': [18.868, 14.457, 12.276, 9.216,  8.763, 8.698,8.264, 7.156]
}

# P 值矩阵（比较 SCC-Net 与其他方法）
p_matrix = np.array([
    # [0.03, 0.04],  # RecoBundles
    # [0.02, 0.03],  # Atlas
    # [0.04, 0.02],  # Atlas_MRtrix
    # [0.01, 0.01],  # TractSeg
    # [0.008, 0.006],  # TransUNet
    # [0.005, 0.002],  # Swin-Unet
    # [0.006, 0.002]   # UCtransNet
    [3.85e-12, 1.2e-12],  # TractSeg
    [9.96e-7, 5.0e-7],  # TransUNet
    [1.10e-6, 6.2e-7],  # Swin-Unet
    [1.8e-4, 1.3e-4]   # UCtransNet
])


# **转置 p 值矩阵**，让 Dice Score 和 HD 横向排列
p_matrix_transposed = p_matrix.T

# ========== 创建画布 ==========
fig = plt.figure(figsize=(6, 7), dpi=300)
gs = gridspec.GridSpec(3, 1, height_ratios=[0.9, 0.9, 0.5], hspace=0.8)

# 颜色设置
colors = plt.cm.Blues(np.linspace(0.4, 1, len(methods)))

# ========== 第一行：Dice Score 柱状图 ==========
ax_dice = plt.subplot(gs[0])
x = np.arange(len(methods))
width = 0.8

bars_dice = ax_dice.bar(x, performance['Dice Score'], width, color=colors, edgecolor='black', linewidth=0.5)

ax_dice.set_ylabel('Dice Score', labelpad=17)
ax_dice.set_ylim(0.6, 0.95)
ax_dice.set_xticks(x)
ax_dice.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax_dice.set_title('A) Dice Score', loc='left', pad=5, fontsize=9, fontweight='bold')

for bar in bars_dice:
    height = bar.get_height()
    ax_dice.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=6)

# ========== 第二行：95% HD 柱状图 ==========
ax_hd = plt.subplot(gs[1])

bars_hd = ax_hd.bar(x, performance['95% HD (mm)'], width, color=colors, edgecolor='black', linewidth=0.5)

ax_hd.set_ylabel('95% HD (mm)', labelpad=17)
ax_hd.set_ylim(0, 20)
ax_hd.set_xticks(x)
ax_hd.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax_hd.set_title('B) 95% Hausdorff Distance', loc='left', pad=5, fontsize=9, fontweight='bold')

for bar in bars_hd:
    height = bar.get_height()
    ax_hd.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f'{height:.3f}', ha='center', va='bottom', fontsize=6)

# ========== 第三行：p值热力图 ==========
# ========== 第三行：p值热力图 ==========

methods = ['TractSeg', 'TransUNet', 'Swin-Unet', 'UCtransNet', r'SC$^{2}$S-Net']



ax_bottom = plt.subplot(gs[2])
methods_short = methods[:-1]  # 排除SAMTractNet

# 创建热力图数据
heatmap_data = pd.DataFrame(p_matrix,
                           index=methods_short,
                           columns=metrics)

# # 绘制热力图
# sns.heatmap(heatmap_data, ax=ax_bottom, cmap='Blues_r',
#            annot=True, fmt=".3f", linewidths=0.5,
#            cbar_kws={'label': 'p-value', 'shrink': 0.8},
#            annot_kws={'size': 7, 'color':'black'})

# 修改 fmt 参数
# heatmap = sns.heatmap(heatmap_data, ax=ax_bottom, cmap='Blues_r',
#            annot=True, fmt=".1e", linewidths=0.5,  # 使用科学计数法
#            cbar_kws={'label': 'p-value', 'shrink': 0.8},
#            annot_kws={'size': 7, 'color': 'black'},
#         #    vmin=0, vmax=0.01
# )
heatmap = sns.heatmap(heatmap_data, ax=ax_bottom, cmap='Blues_r',
            annot=True, fmt=".1e", linewidths=0.5,
            cbar_kws={'label': 'p-value', 'shrink': 0.8},
            annot_kws={'size': 7, 'color': 'red'})  # 设置数字颜色为红色


# cmap=sns.light_palette("blue", as_cmap=True)
for text in ax_bottom.texts:
    p = float(text.get_text())  # 获取 p 值
    if p < 1e-4:
        text.set_color('white')  # p < 0.01 用红色
    else:
        text.set_color('black')  # 其他用黑色


colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([0, 1e-04, 2e-04])  # 设置刻度位置
colorbar.set_ticklabels(['0',  '1.0e-4', '2.0e-4'])  # 设置刻度标签

# sns.heatmap(heatmap_data, cmap=sns.light_palette("blue", as_cmap=True),  
#             annot=True, fmt=".1e", linewidths=0.5,  
#             cbar_kws={'label': 'p-value', 'shrink': 0.8},  
#             annot_kws={'size': 7, 'color': 'black'},  



# 热力图样式调整
ax_bottom.set_xticklabels(ax_bottom.get_xticklabels(), rotation=0,
                        ha='center', rotation_mode='anchor')
ax_bottom.set_yticklabels(ax_bottom.get_yticklabels(),
                        rotation=0, va='center')
ax_bottom.set_xlabel('', labelpad=5)
ax_bottom.set_ylabel('Comparison Methods', labelpad=5)
ax_bottom.set_title('C) Statistical Significance', loc='left', pad=5, fontsize=9, fontweight='bold')

# # 添加显著性标记
# for text in ax_bottom.texts:
#     p = float(text.get_text())
#     if p < 0.001:
#         text.set_text('<0.001***')
#     elif p < 0.01:
#         text.set_text(f'{p:.3f}**')
#     elif p < 0.05:
#         text.set_text(f'{p:.3f}*')
#     else:
#         text.set_text(f'{p:.3f}')

# ========== 全局调整并保存 ==========
plt.tight_layout()
plt.savefig('/home/crq/TractSeg/nature_style_split_plots.pdf', bbox_inches='tight')
plt.show()
