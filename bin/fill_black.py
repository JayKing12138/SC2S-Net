import os
from PIL import Image

def resize_and_fill_images(folder_path):
    # 定义目标尺寸
    target_width = 174
    target_height = 174
    # 定义填充颜色（纯黑色）
    fill_color = (0, 0, 0)

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图像文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    # 打开图像
                    with Image.open(file_path) as img:
                        # 获取原始图像的尺寸
                        width, height = img.size

                        # 创建一个新的纯黑色背景图像
                        new_img = Image.new('RGB', (target_width, target_height), fill_color)

                        # 计算图像放置的位置
                        left = (target_width - width) // 2
                        top = (target_height - height) // 2

                        # 将原始图像粘贴到新图像的中心
                        new_img.paste(img, (left, top))

                        # 保存修改后的图像，覆盖原文件
                        new_img.save(file_path)
                        print(f"已处理: {file_path}")
                except Exception as e:
                    print(f"处理 {file_path} 时出错: {e}")

if __name__ == "__main__":
    # 请将此路径替换为你要处理的文件夹路径
    folder_path = "/data/userdisk1/crq/TractSeg/hcp_exp_5/623844slices"
    resize_and_fill_images(folder_path)