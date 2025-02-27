import numpy as np

# 加载.npz文件
data = np.load('/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_4.pt')

# 查看文件中的所有数组名
print("Arrays in the file:", data.files)

# 访问每个数组并输出内容
for arr_name in data.files:
    print(f"Array '{arr_name}':")
    print(data[arr_name])
