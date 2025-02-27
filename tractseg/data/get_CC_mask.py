import os
import shutil
import nibabel as nib
import numpy as np


def process_folders(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subfolder in os.listdir(input_folder):
        input_subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)

        if os.path.isdir(input_subfolder_path):
            # 创建与输入文件夹结构相同的输出文件夹
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            bundle_masks_file = os.path.join(input_subfolder_path, 'bundle_masks_72.nii.gz')
            mrtrix_peaks_file = os.path.join(input_subfolder_path,'mrtrix_peaks.nii.gz')

            if os.path.isfile(bundle_masks_file):
                # 加载NIfTI文件
                img = nib.load(bundle_masks_file)
                data = img.get_fdata()

                # 提取第46个纤维束（索引从0开始，所以是45）
                fiber_bundle_46 = data[..., 6:13]
                # 将纤维束数据扩展为四维，第四维长度为1
                fiber_bundle_46 = np.expand_dims(fiber_bundle_46, axis=-1)

                # 创建新的NIfTI图像
                new_img = nib.Nifti1Image(fiber_bundle_46, img.affine, img.header)

                # 保存新的NIfTI文件，文件名不变
                nib.save(new_img, os.path.join(output_subfolder_path, 'bundle_masks_72.nii.gz'))

            if os.path.isfile(mrtrix_peaks_file):
                # 复制mrtrix_peaks.nii.gz文件到输出文件夹
                shutil.copy2(mrtrix_peaks_file, output_subfolder_path)


if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = '/home/crq/HCP_preproc2/'
    # 输出文件夹路径
    output_folder = '/home/crq/HCP105_CC_1234567'

    process_folders(input_folder, output_folder)
