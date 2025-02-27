import nibabel as nib
import numpy as np

def pad_nii_image(input_path, output_path):
    # 加载 NIfTI 文件
    nii_img = nib.load(input_path)
    # 获取图像数据
    img_data = nii_img.get_fdata()

    # 目标尺寸
    target_shape = (145, 174, 145)

    # 计算每个维度需要填充的大小
    pad_width = []
    for i in range(min(len(target_shape), img_data.ndim)):
        pad_width_dim = (target_shape[i] - img_data.shape[i]) // 2
        remaining_dim = (target_shape[i] - img_data.shape[i]) % 2
        pad_width.append((pad_width_dim, pad_width_dim + remaining_dim))

    # 如果图像维度大于 3，对多余维度不进行填充
    if img_data.ndim > len(target_shape):
        for _ in range(img_data.ndim - len(target_shape)):
            pad_width.append((0, 0))

    # 填充图像数据
    padded_data = np.pad(img_data, pad_width, mode='constant', constant_values=0)

    # 创建新的 NIfTI 图像对象
    new_nii_img = nib.Nifti1Image(padded_data, nii_img.affine, header=nii_img.header)

    # 保存新的 NIfTI 文件
    nib.save(new_nii_img, output_path)

# 输入和输出文件路径
input_file = '/home/crq/HCP105_6/715041/bundle_masks_72.nii.gz'
output_file = '/home/crq/HCP105_6/715041/final_mask_image.nii.gz'

# 调用函数进行填充和保存
pad_nii_image(input_file, output_file)