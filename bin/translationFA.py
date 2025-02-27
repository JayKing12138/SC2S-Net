import SimpleITK as sitk

if __name__ == "__main__":
    # 原 FA 文件路径
    original_fa_path = "/data/userdisk1/crq/TractSeg/715041/715041FA.nii.gz"
    # 修正后保存的路径
    corrected_fa_path = "/data/userdisk1/crq/TractSeg/715041/715041corrected_FA.nii.gz"

    # 读取 3D FA 图像
    moving_image = sitk.ReadImage(original_fa_path, sitk.sitkFloat32)

    # 定义平移距离（单位：毫米）
    translation_x = -0.9
    translation_y = -4
    translation_z = -21.8

    # 创建平移变换
    translation_transform = sitk.TranslationTransform(3)  # 3D 平移变换
    translation_transform.SetOffset([translation_x, translation_y, translation_z])

    # 应用平移变换到移动图像
    resampled_image = sitk.Resample(moving_image, moving_image, translation_transform, sitk.sitkLinear, 0.0,
                                    moving_image.GetPixelID())

    # 保存平移后的图像
    sitk.WriteImage(resampled_image, corrected_fa_path)