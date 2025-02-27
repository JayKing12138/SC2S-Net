from PIL import Image
import os


def crop_and_resize_images(folder_path):
    target_filename = '3d.png'
    crop_size = (425, 310)
    resize_size = (425, 310)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == target_filename:
                image_path = os.path.join(root, file)
                try:
                    img = Image.open(image_path)
                    width, height = img.size
                    left = (width - crop_size[0]) / 2
                    top = (height - crop_size[1]) / 2
                    right = left + crop_size[0]
                    bottom = top + crop_size[1]
                    cropped_img = img.crop((left, top, right, bottom))
                    resized_img = cropped_img.resize(resize_size)
                    resized_img.save(image_path)
                    print(f'Processed {image_path}')
                except Exception as e:
                    print(f'Error processing {image_path}: {e}')


if __name__ == '__main__':
    folder_path = '/data/userdisk1/crq/TractSeg/hcp_exp_5/623844slices/GT'
    crop_and_resize_images(folder_path)