import nibabel as nib
import numpy as np

def calculate_dice(segmentation_file, mask_file):
    # 读取分割文件
    seg_img = nib.load(segmentation_file)
    seg_data = seg_img.get_fdata()

    # 读取掩码文件
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # 删除 mask 数据的最后一个维度
    if mask_data.ndim > 3:
        mask_data = np.squeeze(mask_data, axis=-1)

    # 确保两个数据的形状相同
    assert seg_data.shape == mask_data.shape, "Segmentation and mask data must have the same shape."

    # 计算交集
    intersection = np.sum(np.logical_and(seg_data > 0, mask_data > 0))

    # 计算分割和掩码的像素总和
    seg_sum = np.sum(seg_data > 0)
    mask_sum = np.sum(mask_data > 0)

    # 计算 Dice 系数
    if seg_sum + mask_sum == 0:
        dice = 1.0  # 如果两者都为空，Dice 系数为 1
    else:
        dice = (2.0 * intersection) / (seg_sum + mask_sum)

    return dice

# 示例使用
segmentation_file = '/data/userdisk1/crq/TractSeg/623844slices/tractseg/segmentations/761957_segmentation.nii.gz'
# segmentation_file = '/data/userdisk1/crq/TractSeg/hcp_exp_5/result/623844_atls/bundle_segmentations/CC.nii.gz'
mask_file = "/home/crq/HCP105_6/761957/bundle_masks_72.nii.gz"

dice_coefficient = calculate_dice(segmentation_file, mask_file)
print(f"Dice coefficient: {dice_coefficient}")
