import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff

def load_nii_gz(file_path):
    """加载.nii.gz文件并返回数据数组"""
    img = nib.load(file_path)
    data = img.get_fdata()
    # 如果数据的最后一个维度是1，则去掉它
    if data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)
    return data

def dice_coefficient(mask1, mask2):
    """计算Dice系数"""
    intersection = np.sum(mask1 * mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    dice = (2.0 * intersection) / (mask1_sum + mask2_sum)
    return dice

def hausdorff_distance_95(mask1, mask2):
    """计算95% Hausdorff距离"""
    # 获取mask1和mask2中非零点的坐标
    points1 = np.argwhere(mask1 > 0)
    points2 = np.argwhere(mask2 > 0)
    
    # 计算双向Hausdorff距离
    dist1 = directed_hausdorff(points1, points2)[0]
    dist2 = directed_hausdorff(points2, points1)[0]
    hausdorff_dist = max(dist1, dist2)
    
    # 计算95% Hausdorff距离
    distances = np.array([np.min(np.linalg.norm(points1 - p, axis=1)) for p in points2])
    hausdorff_95 = np.percentile(distances, 95)
    return hausdorff_95

def main():
    # 加载分割结果和mask文件
    seg_file = '/data/userdisk1/crq/TractSeg/hcp_exp_5/result/623844_recobundles/bundle_segmentations/fixedCC.nii.gz'
    mask_file = "/data/userdisk1/crq/TractSeg/hcp_exp_5/623844slices/GT/final_mask_image.nii.gz"
    
    seg_data = load_nii_gz(seg_file)
    mask_data = load_nii_gz(mask_file)
    
    # 确保seg和mask的形状一致
    assert seg_data.shape == mask_data.shape, "Segmentation and mask shapes do not match!"
    
    # 计算Dice系数
    dice = dice_coefficient(seg_data, mask_data)
    print(f"Dice Coefficient: {dice:.4f}")
    
    # 计算95% Hausdorff距离
    hausdorff_95 = hausdorff_distance_95(seg_data, mask_data)
    print(f"95% Hausdorff Distance: {hausdorff_95:.4f}")

if __name__ == "__main__":
    main()