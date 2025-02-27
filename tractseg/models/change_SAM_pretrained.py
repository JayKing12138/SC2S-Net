##################################
##################################
##################################

#### 查看pth文件 的结构和torch.size()
import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobile_sam.pt"
# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_6.pt"
# pth_file_path = r"/home/crq/segment-anything-main/checkpoints/sam_vit_b_01ec64.pth"
# pth_file_path = r"/home/crq/MedSAM/checkpoint/medsam_vit_b.pth"
# pth_file_path = r"/home/crq/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
pth_file_path = r"/home/crq/sam2/checkpoints/MedSAM2_pretrain.pth"

# pth_file_path = r"/home/crq/MobileSAM/weights/mobile_sam.pt"
# pth_file_path = r"/home/crq/TractSeg/tractseg/models/0mobile_sam.pt"
dict = torch.load(pth_file_path)
for key in dict.values().keys():
    print(key)


# for k,v in dict.items():
#     if 'patch_embed' in k:
#         del dict[k]
# torch.save(dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_6.pt')

# for k,v in dict.items():
#     # if 'patch_embed' in k:
#         # del dict[k]
#     for k,v in v.items():
#         print(k, v.shape)



# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_2.pt"

# # 加载模型权重
# model_dict = torch.load(pth_file_path)

# # 存储需要删除的键
# keys_to_delete = []

# # 遍历字典并找出需要删除的键
# for k in model_dict.keys():
#     if 'patch_embed' in k:
#         keys_to_delete.append(k)

# # 删除需要删除的键
# for k in keys_to_delete:
#     del model_dict[k]

# # 保存修改后的字典
# torch.save(model_dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_6.pt')


    # if k == 'patch_embed.seq.0.c.weight':
    #     print(v.shape)
    # print(k, v.shape)
#################################
#################################
#################################

# 删掉组件里的“image_encoder.”字符

# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only.pt"

# updated_state_dict = {}

# dict = torch.load(pth_file_path)

# for key, value in dict.items():
#     # 替换键中的 "image_encoder" 为 ""（即删除它）
#     new_key = key.replace("image_encoder.", "")
#     updated_state_dict[new_key] = value

# torch.save(updated_state_dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_2.pt')



##################################
##################################
##################################

# 修改pt文件，只保留image encoder部分
# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobile_sam.pt"

# new_state_dict = {}

# dict = torch.load(pth_file_path)

# for k,v in dict.items():
#     if 'image_encoder' in k:
#         new_state_dict[k] = v

# torch.save(new_state_dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only.pt')

####################################
####################################
####################################
# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_2.pt"

# updated_state_dict = {}

# dict = torch.load(pth_file_path)

# new_state_dict = {}

# for k,v in dict.items():
#     if "layers.2.blocks.4" in k or "layers.2.blocks.5" in k or "layers.3.blocks.0" in k or "layers.3.blocks.1" in k or "neck" in k or "head" in k:
#         new_state_dict[k] = v


# torch.save(new_state_dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_3.pt')


#####################################
#####################################
#####################################

# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_4.pt"

# updated_state_dict = {}

# dict = torch.load(pth_file_path)

# new_state_dict = {}

# list_1 = [
#   "layers.2.blocks.4.attn.attention_biases",
#   "layers.2.blocks.4.attn.norm.weight",
#   "layers.2.blocks.4.attn.norm.bias",
#   "layers.2.blocks.4.attn.qkv.weight",
#   "layers.2.blocks.4.attn.qkv.bias",
#   "layers.2.blocks.4.attn.proj.weight",
#   "layers.2.blocks.4.attn.proj.bias",
#   "layers.2.blocks.4.mlp.norm.weight",
#   "layers.2.blocks.4.mlp.norm.bias",
#   "layers.2.blocks.4.mlp.fc1.weight",
#   "layers.2.blocks.4.mlp.fc1.bias",
#   "layers.2.blocks.4.mlp.fc2.weight",
#   "layers.2.blocks.4.mlp.fc2.bias",
#   "layers.2.blocks.4.local_conv.c.weight",
#   "layers.2.blocks.4.local_conv.bn.weight",
#   "layers.2.blocks.4.local_conv.bn.bias",
#   "layers.2.blocks.4.local_conv.bn.running_mean",
#   "layers.2.blocks.4.local_conv.bn.running_var",
#   "layers.2.blocks.4.local_conv.bn.num_batches_tracked",
#   "layers.2.blocks.5.attn.attention_biases",
#   "layers.2.blocks.5.attn.norm.weight",
#   "layers.2.blocks.5.attn.norm.bias",
#   "layers.2.blocks.5.attn.qkv.weight",
#   "layers.2.blocks.5.attn.qkv.bias",
#   "layers.2.blocks.5.attn.proj.weight",
#   "layers.2.blocks.5.attn.proj.bias",
#   "layers.2.blocks.5.mlp.norm.weight",
#   "layers.2.blocks.5.mlp.norm.bias",
#   "layers.2.blocks.5.mlp.fc1.weight",
#   "layers.2.blocks.5.mlp.fc1.bias",
#   "layers.2.blocks.5.mlp.fc2.weight",
#   "layers.2.blocks.5.mlp.fc2.bias",
#   "layers.2.blocks.5.local_conv.c.weight",
#   "layers.2.blocks.5.local_conv.bn.weight",
#   "layers.2.blocks.5.local_conv.bn.bias",
#   "layers.2.blocks.5.local_conv.bn.running_mean",
#   "layers.2.blocks.5.local_conv.bn.running_var",
#   "layers.2.blocks.5.local_conv.bn.num_batches_tracked",
#   "layers.3.blocks.0.attn.attention_biases",
#   "layers.3.blocks.0.attn.norm.weight",
#   "layers.3.blocks.0.attn.norm.bias",
#   "layers.3.blocks.0.attn.qkv.weight",
#   "layers.3.blocks.0.attn.qkv.bias",
#   "layers.3.blocks.0.attn.proj.weight",
#   "layers.3.blocks.0.attn.proj.bias",
#   "layers.3.blocks.0.mlp.norm.weight",
#   "layers.3.blocks.0.mlp.norm.bias",
#   "layers.3.blocks.0.mlp.fc1.weight",
#   "layers.3.blocks.0.mlp.fc1.bias",
#   "layers.3.blocks.0.mlp.fc2.weight",
#   "layers.3.blocks.0.mlp.fc2.bias",
#   "layers.3.blocks.0.local_conv.c.weight",
#   "layers.3.blocks.0.local_conv.bn.weight",
#   "layers.3.blocks.0.local_conv.bn.bias",
#   "layers.3.blocks.0.local_conv.bn.running_mean",
#   "layers.3.blocks.0.local_conv.bn.running_var",
#   "layers.3.blocks.0.local_conv.bn.num_batches_tracked",
#   "layers.3.blocks.1.attn.attention_biases",
#   "layers.3.blocks.1.attn.norm.weight",
#   "layers.3.blocks.1.attn.norm.bias",
#   "layers.3.blocks.1.attn.qkv.weight",
#   "layers.3.blocks.1.attn.qkv.bias",
#   "layers.3.blocks.1.attn.proj.weight",
#   "layers.3.blocks.1.attn.proj.bias",
#   "layers.3.blocks.1.mlp.norm.weight",
#   "layers.3.blocks.1.mlp.norm.bias",
#   "layers.3.blocks.1.mlp.fc1.weight",
#   "layers.3.blocks.1.mlp.fc1.bias",
#   "layers.3.blocks.1.mlp.fc2.weight",
#   "layers.3.blocks.1.mlp.fc2.bias",
#   "layers.3.blocks.1.local_conv.c.weight",
#   "layers.3.blocks.1.local_conv.bn.weight",
#   "layers.3.blocks.1.local_conv.bn.bias",
#   "layers.3.blocks.1.local_conv.bn.running_mean",
#   "layers.3.blocks.1.local_conv.bn.running_var",
#   "layers.3.blocks.1.local_conv.bn.num_batches_tracked",
#   "norm_head.weight",
#   "norm_head.bias",
#   "head.weight",
#   "head.bias",
#   "neck.0.weight",
#   "neck.1.weight",
#   "neck.1.bias",
#   "neck.2.weight",
#   "neck.3.weight",
#   "neck.3.bias"
# ]


# list_2 = [
#   "layers.0.blocks.0.attn.attention_biases",
#   "layers.0.blocks.0.attn.norm.weight",
#   "layers.0.blocks.0.attn.norm.bias",
#   "layers.0.blocks.0.attn.qkv.weight",
#   "layers.0.blocks.0.attn.qkv.bias",
#   "layers.0.blocks.0.attn.proj.weight",
#   "layers.0.blocks.0.attn.proj.bias",
#   "layers.0.blocks.0.mlp.norm.weight",
#   "layers.0.blocks.0.mlp.norm.bias",
#   "layers.0.blocks.0.mlp.fc1.weight",
#   "layers.0.blocks.0.mlp.fc1.bias",
#   "layers.0.blocks.0.mlp.fc2.weight",
#   "layers.0.blocks.0.mlp.fc2.bias",
#   "layers.0.blocks.0.local_conv.c.weight",
#   "layers.0.blocks.0.local_conv.bn.weight",
#   "layers.0.blocks.0.local_conv.bn.bias",
#   "layers.0.blocks.0.local_conv.bn.running_mean",
#   "layers.0.blocks.0.local_conv.bn.running_var",
#   "layers.0.blocks.0.local_conv.bn.num_batches_tracked",
#   "layers.0.blocks.1.attn.attention_biases",
#   "layers.0.blocks.1.attn.norm.weight",
#   "layers.0.blocks.1.attn.norm.bias",
#   "layers.0.blocks.1.attn.qkv.weight",
#   "layers.0.blocks.1.attn.qkv.bias",
#   "layers.0.blocks.1.attn.proj.weight",
#   "layers.0.blocks.1.attn.proj.bias",
#   "layers.0.blocks.1.mlp.norm.weight",
#   "layers.0.blocks.1.mlp.norm.bias",
#   "layers.0.blocks.1.mlp.fc1.weight",
#   "layers.0.blocks.1.mlp.fc1.bias",
#   "layers.0.blocks.1.mlp.fc2.weight",
#   "layers.0.blocks.1.mlp.fc2.bias",
#   "layers.0.blocks.1.local_conv.c.weight",
#   "layers.0.blocks.1.local_conv.bn.weight",
#   "layers.0.blocks.1.local_conv.bn.bias",
#   "layers.0.blocks.1.local_conv.bn.running_mean",
#   "layers.0.blocks.1.local_conv.bn.running_var",
#   "layers.0.blocks.1.local_conv.bn.num_batches_tracked",
#   "layers.1.blocks.0.attn.attention_biases",
#   "layers.1.blocks.0.attn.norm.weight",
#   "layers.1.blocks.0.attn.norm.bias",
#   "layers.1.blocks.0.attn.qkv.weight",
#   "layers.1.blocks.0.attn.qkv.bias",
#   "layers.1.blocks.0.attn.proj.weight",
#   "layers.1.blocks.0.attn.proj.bias",
#   "layers.1.blocks.0.mlp.norm.weight",
#   "layers.1.blocks.0.mlp.norm.bias",
#   "layers.1.blocks.0.mlp.fc1.weight",
#   "layers.1.blocks.0.mlp.fc1.bias",
#   "layers.1.blocks.0.mlp.fc2.weight",
#   "layers.1.blocks.0.mlp.fc2.bias",
#   "layers.1.blocks.0.local_conv.c.weight",
#   "layers.1.blocks.0.local_conv.bn.weight",
#   "layers.1.blocks.0.local_conv.bn.bias",
#   "layers.1.blocks.0.local_conv.bn.running_mean",
#   "layers.1.blocks.0.local_conv.bn.running_var",
#   "layers.1.blocks.0.local_conv.bn.num_batches_tracked",
#   "layers.1.blocks.1.attn.attention_biases",
#   "layers.1.blocks.1.attn.norm.weight",
#   "layers.1.blocks.1.attn.norm.bias",
#   "layers.1.blocks.1.attn.qkv.weight",
#   "layers.1.blocks.1.attn.qkv.bias",
#   "layers.1.blocks.1.attn.proj.weight",
#   "layers.1.blocks.1.attn.proj.bias",
#   "layers.1.blocks.1.mlp.norm.weight",
#   "layers.1.blocks.1.mlp.norm.bias",
#   "layers.1.blocks.1.mlp.fc1.weight",
#   "layers.1.blocks.1.mlp.fc1.bias",
#   "layers.1.blocks.1.mlp.fc2.weight",
#   "layers.1.blocks.1.mlp.fc2.bias",
#   "layers.1.blocks.1.local_conv.c.weight",
#   "layers.1.blocks.1.local_conv.bn.weight",
#   "layers.1.blocks.1.local_conv.bn.bias",
#   "layers.1.blocks.1.local_conv.bn.running_mean",
#   "layers.1.blocks.1.local_conv.bn.running_var",
#   "layers.1.blocks.1.local_conv.bn.num_batches_tracked",
#   "norm_head.weight",
#   "norm_head.bias",
#   "head.weight",
#   "head.bias",
#   "neck.0.weight",
#   "neck.1.weight",
#   "neck.1.bias",
#   "neck.2.weight",
#   "neck.3.weight",
#   "neck.3.bias"
# ]

# new_keys = [
#     "patch_embed.seq.0.c.weight",
#     "patch_embed.seq.0.bn.weight",
#     "patch_embed.seq.0.bn.bias",
#     "patch_embed.seq.0.bn.running_mean",
#     "patch_embed.seq.0.bn.running_var",
#     "patch_embed.seq.0.bn.num_batches_tracked",
#     "patch_embed.seq.2.c.weight",
#     "patch_embed.seq.2.bn.weight",
#     "patch_embed.seq.2.bn.bias",
#     "patch_embed.seq.2.bn.running_mean",
#     "patch_embed.seq.2.bn.running_var",
#     "patch_embed.seq.2.bn.num_batches_tracked",
# ]

# # 创建一个新的state_dict，用于添加新键
# new_state_dict = {key: None for key in new_keys}

# # 合并新的键和现有的参数
# updated_state_dict = {**new_state_dict, **dict}

# # for i in list_1,list_2:
# #     for k ,v in dict.items():
# #         if "layers.2.blocks.4" in k or "layers.2.blocks.5" in k or "layers.3.blocks.0" in k or "layers.3.blocks.1" in k or "neck" in k or "head" in k:
# #             new_state_dict[k] = v


# # for old_key, new_key in zip(list_1, list_2):
# #     if old_key in dict:
# #         new_state_dict[new_key] = dict.pop(old_key)

# torch.save(updated_state_dict, r'/home/crq/MobileSAM/weights/mobilesam_image_encoder_only_5.pt')

####################################
####################################
####################################

########### 查看pt的patch_embed的具体数值

# image_encoder_patch_embed = [
#     "image_encoder.patch_embed.seq.0.c.weight",
#     "image_encoder.patch_embed.seq.0.bn.weight",
#     "image_encoder.patch_embed.seq.0.bn.bias",
#     "image_encoder.patch_embed.seq.0.bn.running_mean",
#     "image_encoder.patch_embed.seq.0.bn.running_var",
#     "image_encoder.patch_embed.seq.0.bn.num_batches_tracked",
#     "image_encoder.patch_embed.seq.2.c.weight",
#     "image_encoder.patch_embed.seq.2.bn.weight",
#     "image_encoder.patch_embed.seq.2.bn.bias",
#     "image_encoder.patch_embed.seq.2.bn.running_mean",
#     "image_encoder.patch_embed.seq.2.bn.running_var",
#     "image_encoder.patch_embed.seq.2.bn.num_batches_tracked"
# ]

# import torch

# pth_file_path = r"/home/crq/MobileSAM/weights/mobilesam_image_encoder_only.pt"
# # pth_file_path = r"/home/crq/MobileSAM/weights/mobile_sam.pt"
# # pth_file_path = r"/home/crq/TractSeg/tractseg/models/1mobile_sam.pt"
# dict = torch.load(pth_file_path)

# for i in image_encoder_patch_embed:
#     print(dict[i])

#############################################
#############################################
#############################################



# # torch.save(dict, "/home/crq/TractSeg/tractseg/models/1mobile_sam.pt")






# dict["image_encoder.patch_embed.seq.0.c.weight"]=torch.ones((32, 9, 3, 3))
# # del dict["image_encoder.head.weight"]
# torch.save(dict, '/home/crq/TractSeg/tractseg/models/mobile_sam_1.pt')

# # for name,param in sam_image_encoder.named_parameters():
# # 	if (name == "patch_embed.proj.weight"):
# # 		print(name,param.shape)

# # print(dict["image_encoder.head.bias"])
# # print(dict["image_encoder.patch_embed.proj.weight"].shape)
# # # torch.Size([768, 3, 16, 16])
# # dict["image_encoder.patch_embed.proj.weight"] = torch.ones((768, 9, 16, 16))

# # torch.save(dict, '/home/crq/SAMIHS-main/pretrained/sam_vit_b_01ec64.pth')

# # image_encoder.patch_embed.seq.0.c.weight /// torch.Size([32, 3, 3, 3])
# # image_encoder.patch_embed.seq.0.bn.weight
# # image_encoder.patch_embed.seq.0.bn.bias
# # image_encoder.patch_embed.seq.0.bn.running_mean
# # image_encoder.patch_embed.seq.0.bn.running_var
# # image_encoder.patch_embed.seq.0.bn.num_batches_tracked
# # image_encoder.patch_embed.seq.2.c.weight /// torch.Size([64, 32, 3, 3])
# # image_encoder.patch_embed.seq.2.bn.weight
# # image_encoder.patch_embed.seq.2.bn.bias
# # image_encoder.patch_embed.seq.2.bn.running_mean
# # image_encoder.patch_embed.seq.2.bn.running_var
# # image_encoder.patch_embed.seq.2.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.0.conv1.c.weight
# # image_encoder.layers.0.blocks.0.conv1.bn.weight
# # image_encoder.layers.0.blocks.0.conv1.bn.bias
# # image_encoder.layers.0.blocks.0.conv1.bn.running_mean
# # image_encoder.layers.0.blocks.0.conv1.bn.running_var
# # image_encoder.layers.0.blocks.0.conv1.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.0.conv2.c.weight
# # image_encoder.layers.0.blocks.0.conv2.bn.weight
# # image_encoder.layers.0.blocks.0.conv2.bn.bias
# # image_encoder.layers.0.blocks.0.conv2.bn.running_mean
# # image_encoder.layers.0.blocks.0.conv2.bn.running_var
# # image_encoder.layers.0.blocks.0.conv2.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.0.conv3.c.weight
# # image_encoder.layers.0.blocks.0.conv3.bn.weight
# # image_encoder.layers.0.blocks.0.conv3.bn.bias
# # image_encoder.layers.0.blocks.0.conv3.bn.running_mean
# # image_encoder.layers.0.blocks.0.conv3.bn.running_var
# # image_encoder.layers.0.blocks.0.conv3.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.1.conv1.c.weight
# # image_encoder.layers.0.blocks.1.conv1.bn.weight
# # image_encoder.layers.0.blocks.1.conv1.bn.bias
# # image_encoder.layers.0.blocks.1.conv1.bn.running_mean
# # image_encoder.layers.0.blocks.1.conv1.bn.running_var
# # image_encoder.layers.0.blocks.1.conv1.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.1.conv2.c.weight
# # image_encoder.layers.0.blocks.1.conv2.bn.weight
# # image_encoder.layers.0.blocks.1.conv2.bn.bias
# # image_encoder.layers.0.blocks.1.conv2.bn.running_mean
# # image_encoder.layers.0.blocks.1.conv2.bn.running_var
# # image_encoder.layers.0.blocks.1.conv2.bn.num_batches_tracked
# # image_encoder.layers.0.blocks.1.conv3.c.weight
# # image_encoder.layers.0.blocks.1.conv3.bn.weight
# # image_encoder.layers.0.blocks.1.conv3.bn.bias
# # image_encoder.layers.0.blocks.1.conv3.bn.running_mean
# # image_encoder.layers.0.blocks.1.conv3.bn.running_var
# # image_encoder.layers.0.blocks.1.conv3.bn.num_batches_tracked
# # image_encoder.layers.0.downsample.conv1.c.weight
# # image_encoder.layers.0.downsample.conv1.bn.weight
# # image_encoder.layers.0.downsample.conv1.bn.bias
# # image_encoder.layers.0.downsample.conv1.bn.running_mean
# # image_encoder.layers.0.downsample.conv1.bn.running_var
# # image_encoder.layers.0.downsample.conv1.bn.num_batches_tracked
# # image_encoder.layers.0.downsample.conv2.c.weight
# # image_encoder.layers.0.downsample.conv2.bn.weight
# # image_encoder.layers.0.downsample.conv2.bn.bias
# # image_encoder.layers.0.downsample.conv2.bn.running_mean
# # image_encoder.layers.0.downsample.conv2.bn.running_var
# # image_encoder.layers.0.downsample.conv2.bn.num_batches_tracked
# # image_encoder.layers.0.downsample.conv3.c.weight
# # image_encoder.layers.0.downsample.conv3.bn.weight
# # image_encoder.layers.0.downsample.conv3.bn.bias
# # image_encoder.layers.0.downsample.conv3.bn.running_mean
# # image_encoder.layers.0.downsample.conv3.bn.running_var
# # image_encoder.layers.0.downsample.conv3.bn.num_batches_tracked
# # image_encoder.layers.1.blocks.0.attn.attention_biases
# # image_encoder.layers.1.blocks.0.attn.norm.weight
# # image_encoder.layers.1.blocks.0.attn.norm.bias
# # image_encoder.layers.1.blocks.0.attn.qkv.weight
# # image_encoder.layers.1.blocks.0.attn.qkv.bias
# # image_encoder.layers.1.blocks.0.attn.proj.weight
# # image_encoder.layers.1.blocks.0.attn.proj.bias
# # image_encoder.layers.1.blocks.0.mlp.norm.weight
# # image_encoder.layers.1.blocks.0.mlp.norm.bias
# # image_encoder.layers.1.blocks.0.mlp.fc1.weight
# # image_encoder.layers.1.blocks.0.mlp.fc1.bias
# # image_encoder.layers.1.blocks.0.mlp.fc2.weight
# # image_encoder.layers.1.blocks.0.mlp.fc2.bias
# # image_encoder.layers.1.blocks.0.local_conv.c.weight
# # image_encoder.layers.1.blocks.0.local_conv.bn.weight
# # image_encoder.layers.1.blocks.0.local_conv.bn.bias
# # image_encoder.layers.1.blocks.0.local_conv.bn.running_mean
# # image_encoder.layers.1.blocks.0.local_conv.bn.running_var
# # image_encoder.layers.1.blocks.0.local_conv.bn.num_batches_tracked
# # image_encoder.layers.1.blocks.1.attn.attention_biases
# # image_encoder.layers.1.blocks.1.attn.norm.weight
# # image_encoder.layers.1.blocks.1.attn.norm.bias
# # image_encoder.layers.1.blocks.1.attn.qkv.weight
# # image_encoder.layers.1.blocks.1.attn.qkv.bias
# # image_encoder.layers.1.blocks.1.attn.proj.weight
# # image_encoder.layers.1.blocks.1.attn.proj.bias
# # image_encoder.layers.1.blocks.1.mlp.norm.weight
# # image_encoder.layers.1.blocks.1.mlp.norm.bias
# # image_encoder.layers.1.blocks.1.mlp.fc1.weight
# # image_encoder.layers.1.blocks.1.mlp.fc1.bias
# # image_encoder.layers.1.blocks.1.mlp.fc2.weight
# # image_encoder.layers.1.blocks.1.mlp.fc2.bias
# # image_encoder.layers.1.blocks.1.local_conv.c.weight
# # image_encoder.layers.1.blocks.1.local_conv.bn.weight
# # image_encoder.layers.1.blocks.1.local_conv.bn.bias
# # image_encoder.layers.1.blocks.1.local_conv.bn.running_mean
# # image_encoder.layers.1.blocks.1.local_conv.bn.running_var
# # image_encoder.layers.1.blocks.1.local_conv.bn.num_batches_tracked
# # image_encoder.layers.1.downsample.conv1.c.weight
# # image_encoder.layers.1.downsample.conv1.bn.weight
# # image_encoder.layers.1.downsample.conv1.bn.bias
# # image_encoder.layers.1.downsample.conv1.bn.running_mean
# # image_encoder.layers.1.downsample.conv1.bn.running_var
# # image_encoder.layers.1.downsample.conv1.bn.num_batches_tracked
# # image_encoder.layers.1.downsample.conv2.c.weight
# # image_encoder.layers.1.downsample.conv2.bn.weight
# # image_encoder.layers.1.downsample.conv2.bn.bias
# # image_encoder.layers.1.downsample.conv2.bn.running_mean
# # image_encoder.layers.1.downsample.conv2.bn.running_var
# # image_encoder.layers.1.downsample.conv2.bn.num_batches_tracked
# # image_encoder.layers.1.downsample.conv3.c.weight
# # image_encoder.layers.1.downsample.conv3.bn.weight
# # image_encoder.layers.1.downsample.conv3.bn.bias
# # image_encoder.layers.1.downsample.conv3.bn.running_mean
# # image_encoder.layers.1.downsample.conv3.bn.running_var
# # image_encoder.layers.1.downsample.conv3.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.0.attn.attention_biases
# # image_encoder.layers.2.blocks.0.attn.norm.weight
# # image_encoder.layers.2.blocks.0.attn.norm.bias
# # image_encoder.layers.2.blocks.0.attn.qkv.weight
# # image_encoder.layers.2.blocks.0.attn.qkv.bias
# # image_encoder.layers.2.blocks.0.attn.proj.weight
# # image_encoder.layers.2.blocks.0.attn.proj.bias
# # image_encoder.layers.2.blocks.0.mlp.norm.weight
# # image_encoder.layers.2.blocks.0.mlp.norm.bias
# # image_encoder.layers.2.blocks.0.mlp.fc1.weight
# # image_encoder.layers.2.blocks.0.mlp.fc1.bias
# # image_encoder.layers.2.blocks.0.mlp.fc2.weight
# # image_encoder.layers.2.blocks.0.mlp.fc2.bias
# # image_encoder.layers.2.blocks.0.local_conv.c.weight
# # image_encoder.layers.2.blocks.0.local_conv.bn.weight
# # image_encoder.layers.2.blocks.0.local_conv.bn.bias
# # image_encoder.layers.2.blocks.0.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.0.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.0.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.1.attn.attention_biases
# # image_encoder.layers.2.blocks.1.attn.norm.weight
# # image_encoder.layers.2.blocks.1.attn.norm.bias
# # image_encoder.layers.2.blocks.1.attn.qkv.weight
# # image_encoder.layers.2.blocks.1.attn.qkv.bias
# # image_encoder.layers.2.blocks.1.attn.proj.weight
# # image_encoder.layers.2.blocks.1.attn.proj.bias
# # image_encoder.layers.2.blocks.1.mlp.norm.weight
# # image_encoder.layers.2.blocks.1.mlp.norm.bias
# # image_encoder.layers.2.blocks.1.mlp.fc1.weight
# # image_encoder.layers.2.blocks.1.mlp.fc1.bias
# # image_encoder.layers.2.blocks.1.mlp.fc2.weight
# # image_encoder.layers.2.blocks.1.mlp.fc2.bias
# # image_encoder.layers.2.blocks.1.local_conv.c.weight
# # image_encoder.layers.2.blocks.1.local_conv.bn.weight
# # image_encoder.layers.2.blocks.1.local_conv.bn.bias
# # image_encoder.layers.2.blocks.1.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.1.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.1.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.2.attn.attention_biases
# # image_encoder.layers.2.blocks.2.attn.norm.weight
# # image_encoder.layers.2.blocks.2.attn.norm.bias
# # image_encoder.layers.2.blocks.2.attn.qkv.weight
# # image_encoder.layers.2.blocks.2.attn.qkv.bias
# # image_encoder.layers.2.blocks.2.attn.proj.weight
# # image_encoder.layers.2.blocks.2.attn.proj.bias
# # image_encoder.layers.2.blocks.2.mlp.norm.weight
# # image_encoder.layers.2.blocks.2.mlp.norm.bias
# # image_encoder.layers.2.blocks.2.mlp.fc1.weight
# # image_encoder.layers.2.blocks.2.mlp.fc1.bias
# # image_encoder.layers.2.blocks.2.mlp.fc2.weight
# # image_encoder.layers.2.blocks.2.mlp.fc2.bias
# # image_encoder.layers.2.blocks.2.local_conv.c.weight
# # image_encoder.layers.2.blocks.2.local_conv.bn.weight
# # image_encoder.layers.2.blocks.2.local_conv.bn.bias
# # image_encoder.layers.2.blocks.2.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.2.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.2.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.3.attn.attention_biases
# # image_encoder.layers.2.blocks.3.attn.norm.weight
# # image_encoder.layers.2.blocks.3.attn.norm.bias
# # image_encoder.layers.2.blocks.3.attn.qkv.weight
# # image_encoder.layers.2.blocks.3.attn.qkv.bias
# # image_encoder.layers.2.blocks.3.attn.proj.weight
# # image_encoder.layers.2.blocks.3.attn.proj.bias
# # image_encoder.layers.2.blocks.3.mlp.norm.weight
# # image_encoder.layers.2.blocks.3.mlp.norm.bias
# # image_encoder.layers.2.blocks.3.mlp.fc1.weight
# # image_encoder.layers.2.blocks.3.mlp.fc1.bias
# # image_encoder.layers.2.blocks.3.mlp.fc2.weight
# # image_encoder.layers.2.blocks.3.mlp.fc2.bias
# # image_encoder.layers.2.blocks.3.local_conv.c.weight
# # image_encoder.layers.2.blocks.3.local_conv.bn.weight
# # image_encoder.layers.2.blocks.3.local_conv.bn.bias
# # image_encoder.layers.2.blocks.3.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.3.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.3.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.4.attn.attention_biases
# # image_encoder.layers.2.blocks.4.attn.norm.weight
# # image_encoder.layers.2.blocks.4.attn.norm.bias
# # image_encoder.layers.2.blocks.4.attn.qkv.weight
# # image_encoder.layers.2.blocks.4.attn.qkv.bias
# # image_encoder.layers.2.blocks.4.attn.proj.weight
# # image_encoder.layers.2.blocks.4.attn.proj.bias
# # image_encoder.layers.2.blocks.4.mlp.norm.weight
# # image_encoder.layers.2.blocks.4.mlp.norm.bias
# # image_encoder.layers.2.blocks.4.mlp.fc1.weight
# # image_encoder.layers.2.blocks.4.mlp.fc1.bias
# # image_encoder.layers.2.blocks.4.mlp.fc2.weight
# # image_encoder.layers.2.blocks.4.mlp.fc2.bias
# # image_encoder.layers.2.blocks.4.local_conv.c.weight
# # image_encoder.layers.2.blocks.4.local_conv.bn.weight
# # image_encoder.layers.2.blocks.4.local_conv.bn.bias
# # image_encoder.layers.2.blocks.4.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.4.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.4.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.blocks.5.attn.attention_biases
# # image_encoder.layers.2.blocks.5.attn.norm.weight
# # image_encoder.layers.2.blocks.5.attn.norm.bias
# # image_encoder.layers.2.blocks.5.attn.qkv.weight
# # image_encoder.layers.2.blocks.5.attn.qkv.bias
# # image_encoder.layers.2.blocks.5.attn.proj.weight
# # image_encoder.layers.2.blocks.5.attn.proj.bias
# # image_encoder.layers.2.blocks.5.mlp.norm.weight
# # image_encoder.layers.2.blocks.5.mlp.norm.bias
# # image_encoder.layers.2.blocks.5.mlp.fc1.weight
# # image_encoder.layers.2.blocks.5.mlp.fc1.bias
# # image_encoder.layers.2.blocks.5.mlp.fc2.weight
# # image_encoder.layers.2.blocks.5.mlp.fc2.bias
# # image_encoder.layers.2.blocks.5.local_conv.c.weight
# # image_encoder.layers.2.blocks.5.local_conv.bn.weight
# # image_encoder.layers.2.blocks.5.local_conv.bn.bias
# # image_encoder.layers.2.blocks.5.local_conv.bn.running_mean
# # image_encoder.layers.2.blocks.5.local_conv.bn.running_var
# # image_encoder.layers.2.blocks.5.local_conv.bn.num_batches_tracked
# # image_encoder.layers.2.downsample.conv1.c.weight
# # image_encoder.layers.2.downsample.conv1.bn.weight
# # image_encoder.layers.2.downsample.conv1.bn.bias
# # image_encoder.layers.2.downsample.conv1.bn.running_mean
# # image_encoder.layers.2.downsample.conv1.bn.running_var
# # image_encoder.layers.2.downsample.conv1.bn.num_batches_tracked
# # image_encoder.layers.2.downsample.conv2.c.weight
# # image_encoder.layers.2.downsample.conv2.bn.weight
# # image_encoder.layers.2.downsample.conv2.bn.bias
# # image_encoder.layers.2.downsample.conv2.bn.running_mean
# # image_encoder.layers.2.downsample.conv2.bn.running_var
# # image_encoder.layers.2.downsample.conv2.bn.num_batches_tracked
# # image_encoder.layers.2.downsample.conv3.c.weight
# # image_encoder.layers.2.downsample.conv3.bn.weight
# # image_encoder.layers.2.downsample.conv3.bn.bias
# # image_encoder.layers.2.downsample.conv3.bn.running_mean
# # image_encoder.layers.2.downsample.conv3.bn.running_var
# # image_encoder.layers.2.downsample.conv3.bn.num_batches_tracked
# # image_encoder.layers.3.blocks.0.attn.attention_biases
# # image_encoder.layers.3.blocks.0.attn.norm.weight
# # image_encoder.layers.3.blocks.0.attn.norm.bias
# # image_encoder.layers.3.blocks.0.attn.qkv.weight
# # image_encoder.layers.3.blocks.0.attn.qkv.bias
# # image_encoder.layers.3.blocks.0.attn.proj.weight
# # image_encoder.layers.3.blocks.0.attn.proj.bias
# # image_encoder.layers.3.blocks.0.mlp.norm.weight
# # image_encoder.layers.3.blocks.0.mlp.norm.bias
# # image_encoder.layers.3.blocks.0.mlp.fc1.weight
# # image_encoder.layers.3.blocks.0.mlp.fc1.bias
# # image_encoder.layers.3.blocks.0.mlp.fc2.weight
# # image_encoder.layers.3.blocks.0.mlp.fc2.bias
# # image_encoder.layers.3.blocks.0.local_conv.c.weight
# # image_encoder.layers.3.blocks.0.local_conv.bn.weight
# # image_encoder.layers.3.blocks.0.local_conv.bn.bias
# # image_encoder.layers.3.blocks.0.local_conv.bn.running_mean
# # image_encoder.layers.3.blocks.0.local_conv.bn.running_var
# # image_encoder.layers.3.blocks.0.local_conv.bn.num_batches_tracked
# # image_encoder.layers.3.blocks.1.attn.attention_biases
# # image_encoder.layers.3.blocks.1.attn.norm.weight
# # image_encoder.layers.3.blocks.1.attn.norm.bias
# # image_encoder.layers.3.blocks.1.attn.qkv.weight
# # image_encoder.layers.3.blocks.1.attn.qkv.bias
# # image_encoder.layers.3.blocks.1.attn.proj.weight
# # image_encoder.layers.3.blocks.1.attn.proj.bias
# # image_encoder.layers.3.blocks.1.mlp.norm.weight
# # image_encoder.layers.3.blocks.1.mlp.norm.bias
# # image_encoder.layers.3.blocks.1.mlp.fc1.weight
# # image_encoder.layers.3.blocks.1.mlp.fc1.bias
# # image_encoder.layers.3.blocks.1.mlp.fc2.weight
# # image_encoder.layers.3.blocks.1.mlp.fc2.bias
# # image_encoder.layers.3.blocks.1.local_conv.c.weight
# # image_encoder.layers.3.blocks.1.local_conv.bn.weight
# # image_encoder.layers.3.blocks.1.local_conv.bn.bias
# # image_encoder.layers.3.blocks.1.local_conv.bn.running_mean
# # image_encoder.layers.3.blocks.1.local_conv.bn.running_var
# # image_encoder.layers.3.blocks.1.local_conv.bn.num_batches_tracked
# # image_encoder.norm_head.weight
# # image_encoder.norm_head.bias
# # image_encoder.head.weight
# # image_encoder.head.bias
# # image_encoder.neck.0.weight
# # image_encoder.neck.1.weight
# # image_encoder.neck.1.bias
# # image_encoder.neck.2.weight
# # image_encoder.neck.3.weight
# # image_encoder.neck.3.bias



# # print(dict["image_encoder.patch_embed.proj.weight"].shape)

# # neck.0.weight torch.Size([256, 768, 1, 1])
# # neck.1.weight torch.Size([256])
# # neck.1.bias torch.Size([256])
# # neck.2.weight torch.Size([256, 256, 3, 3])
# # neck.3.weight torch.Size([256])
# # neck.3.bias torch.Size([256])

# # dict["image_encoder.neck.0.weight"]=torch.ones((1024,768,1,1))
# # dict["image_encoder.neck.1.weight"]=torch.ones(1024)
# # dict["image_encoder.neck.1.bias"]=torch.ones(1024)
# # dict["image_encoder.neck.2.weight"]=torch.ones((1024,1024,1,1))
# # dict["image_encoder.neck.3.weight"]=torch.ones(1024)
# # dict["image_encoder.neck.3.bias"]=torch.ones(1024)


# # torch.save(dict, '/home/crq/SAMIHS-main/pretrained/sam_vit_b_01ec64.pth')

# # print(dict["image_encoder.neck.1.weight"].shape)

# # pos_embed torch.Size([1, 64, 64, 768])
# # patch_embed.proj.weight torch.Size([768, 3, 16, 16])
# # patch_embed.proj.bias torch.Size([768])
# # blocks.0.norm1.weight torch.Size([768])
# # blocks.0.norm1.bias torch.Size([768])
# # blocks.0.attn.rel_pos_h torch.Size([27, 64])
# # blocks.0.attn.rel_pos_w torch.Size([27, 64])
# # blocks.0.attn.qkv.weight torch.Size([2304, 768])
# # blocks.0.attn.qkv.bias torch.Size([2304])
# # blocks.0.attn.proj.weight torch.Size([768, 768])
# # blocks.0.attn.proj.bias torch.Size([768])
# # blocks.0.norm2.weight torch.Size([768])
# # blocks.0.norm2.bias torch.Size([768])
# # blocks.0.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.0.mlp.lin1.bias torch.Size([3072])
# # blocks.0.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.0.mlp.lin2.bias torch.Size([768])
# # blocks.1.norm1.weight torch.Size([768])
# # blocks.1.norm1.bias torch.Size([768])
# # blocks.1.attn.rel_pos_h torch.Size([27, 64])
# # blocks.1.attn.rel_pos_w torch.Size([27, 64])
# # blocks.1.attn.qkv.weight torch.Size([2304, 768])
# # blocks.1.attn.qkv.bias torch.Size([2304])
# # blocks.1.attn.proj.weight torch.Size([768, 768])
# # blocks.1.attn.proj.bias torch.Size([768])
# # blocks.1.norm2.weight torch.Size([768])
# # blocks.1.norm2.bias torch.Size([768])
# # blocks.1.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.1.mlp.lin1.bias torch.Size([3072])
# # blocks.1.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.1.mlp.lin2.bias torch.Size([768])
# # blocks.2.norm1.weight torch.Size([768])
# # blocks.2.norm1.bias torch.Size([768])
# # blocks.2.attn.rel_pos_h torch.Size([127, 64])
# # blocks.2.attn.rel_pos_w torch.Size([127, 64])
# # blocks.2.attn.qkv.weight torch.Size([2304, 768])
# # blocks.2.attn.qkv.bias torch.Size([2304])
# # blocks.2.attn.proj.weight torch.Size([768, 768])
# # blocks.2.attn.proj.bias torch.Size([768])
# # blocks.2.norm2.weight torch.Size([768])
# # blocks.2.norm2.bias torch.Size([768])
# # blocks.2.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.2.mlp.lin1.bias torch.Size([3072])
# # blocks.2.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.2.mlp.lin2.bias torch.Size([768])
# # blocks.3.norm1.weight torch.Size([768])
# # blocks.3.norm1.bias torch.Size([768])
# # blocks.3.attn.rel_pos_h torch.Size([27, 64])
# # blocks.3.attn.rel_pos_w torch.Size([27, 64])
# # blocks.3.attn.qkv.weight torch.Size([2304, 768])
# # blocks.3.attn.qkv.bias torch.Size([2304])
# # blocks.3.attn.proj.weight torch.Size([768, 768])
# # blocks.3.attn.proj.bias torch.Size([768])
# # blocks.3.norm2.weight torch.Size([768])
# # blocks.3.norm2.bias torch.Size([768])
# # blocks.3.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.3.mlp.lin1.bias torch.Size([3072])
# # blocks.3.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.3.mlp.lin2.bias torch.Size([768])
# # blocks.4.norm1.weight torch.Size([768])
# # blocks.4.norm1.bias torch.Size([768])
# # blocks.4.attn.rel_pos_h torch.Size([27, 64])
# # blocks.4.attn.rel_pos_w torch.Size([27, 64])
# # blocks.4.attn.qkv.weight torch.Size([2304, 768])
# # blocks.4.attn.qkv.bias torch.Size([2304])
# # blocks.4.attn.proj.weight torch.Size([768, 768])
# # blocks.4.attn.proj.bias torch.Size([768])
# # blocks.4.norm2.weight torch.Size([768])
# # blocks.4.norm2.bias torch.Size([768])
# # blocks.4.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.4.mlp.lin1.bias torch.Size([3072])
# # blocks.4.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.4.mlp.lin2.bias torch.Size([768])
# # blocks.5.norm1.weight torch.Size([768])
# # blocks.5.norm1.bias torch.Size([768])
# # blocks.5.attn.rel_pos_h torch.Size([127, 64])
# # blocks.5.attn.rel_pos_w torch.Size([127, 64])
# # blocks.5.attn.qkv.weight torch.Size([2304, 768])
# # blocks.5.attn.qkv.bias torch.Size([2304])
# # blocks.5.attn.proj.weight torch.Size([768, 768])
# # blocks.5.attn.proj.bias torch.Size([768])
# # blocks.5.norm2.weight torch.Size([768])
# # blocks.5.norm2.bias torch.Size([768])
# # blocks.5.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.5.mlp.lin1.bias torch.Size([3072])
# # blocks.5.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.5.mlp.lin2.bias torch.Size([768])
# # blocks.6.norm1.weight torch.Size([768])
# # blocks.6.norm1.bias torch.Size([768])
# # blocks.6.attn.rel_pos_h torch.Size([27, 64])
# # blocks.6.attn.rel_pos_w torch.Size([27, 64])
# # blocks.6.attn.qkv.weight torch.Size([2304, 768])
# # blocks.6.attn.qkv.bias torch.Size([2304])
# # blocks.6.attn.proj.weight torch.Size([768, 768])
# # blocks.6.attn.proj.bias torch.Size([768])
# # blocks.6.norm2.weight torch.Size([768])
# # blocks.6.norm2.bias torch.Size([768])
# # blocks.6.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.6.mlp.lin1.bias torch.Size([3072])
# # blocks.6.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.6.mlp.lin2.bias torch.Size([768])
# # blocks.7.norm1.weight torch.Size([768])
# # blocks.7.norm1.bias torch.Size([768])
# # blocks.7.attn.rel_pos_h torch.Size([27, 64])
# # blocks.7.attn.rel_pos_w torch.Size([27, 64])
# # blocks.7.attn.qkv.weight torch.Size([2304, 768])
# # blocks.7.attn.qkv.bias torch.Size([2304])
# # blocks.7.attn.proj.weight torch.Size([768, 768])
# # blocks.7.attn.proj.bias torch.Size([768])
# # blocks.7.norm2.weight torch.Size([768])
# # blocks.7.norm2.bias torch.Size([768])
# # blocks.7.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.7.mlp.lin1.bias torch.Size([3072])
# # blocks.7.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.7.mlp.lin2.bias torch.Size([768])
# # blocks.8.norm1.weight torch.Size([768])
# # blocks.8.norm1.bias torch.Size([768])
# # blocks.8.attn.rel_pos_h torch.Size([127, 64])
# # blocks.8.attn.rel_pos_w torch.Size([127, 64])
# # blocks.8.attn.qkv.weight torch.Size([2304, 768])
# # blocks.8.attn.qkv.bias torch.Size([2304])
# # blocks.8.attn.proj.weight torch.Size([768, 768])
# # blocks.8.attn.proj.bias torch.Size([768])
# # blocks.8.norm2.weight torch.Size([768])
# # blocks.8.norm2.bias torch.Size([768])
# # blocks.8.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.8.mlp.lin1.bias torch.Size([3072])
# # blocks.8.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.8.mlp.lin2.bias torch.Size([768])
# # blocks.9.norm1.weight torch.Size([768])
# # blocks.9.norm1.bias torch.Size([768])
# # blocks.9.attn.rel_pos_h torch.Size([27, 64])
# # blocks.9.attn.rel_pos_w torch.Size([27, 64])
# # blocks.9.attn.qkv.weight torch.Size([2304, 768])
# # blocks.9.attn.qkv.bias torch.Size([2304])
# # blocks.9.attn.proj.weight torch.Size([768, 768])
# # blocks.9.attn.proj.bias torch.Size([768])
# # blocks.9.norm2.weight torch.Size([768])
# # blocks.9.norm2.bias torch.Size([768])
# # blocks.9.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.9.mlp.lin1.bias torch.Size([3072])
# # blocks.9.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.9.mlp.lin2.bias torch.Size([768])
# # blocks.10.norm1.weight torch.Size([768])
# # blocks.10.norm1.bias torch.Size([768])
# # blocks.10.attn.rel_pos_h torch.Size([27, 64])
# # blocks.10.attn.rel_pos_w torch.Size([27, 64])
# # blocks.10.attn.qkv.weight torch.Size([2304, 768])
# # blocks.10.attn.qkv.bias torch.Size([2304])
# # blocks.10.attn.proj.weight torch.Size([768, 768])
# # blocks.10.attn.proj.bias torch.Size([768])
# # blocks.10.norm2.weight torch.Size([768])
# # blocks.10.norm2.bias torch.Size([768])
# # blocks.10.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.10.mlp.lin1.bias torch.Size([3072])
# # blocks.10.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.10.mlp.lin2.bias torch.Size([768])
# # blocks.11.norm1.weight torch.Size([768])
# # blocks.11.norm1.bias torch.Size([768])
# # blocks.11.attn.rel_pos_h torch.Size([127, 64])
# # blocks.11.attn.rel_pos_w torch.Size([127, 64])
# # blocks.11.attn.qkv.weight torch.Size([2304, 768])
# # blocks.11.attn.qkv.bias torch.Size([2304])
# # blocks.11.attn.proj.weight torch.Size([768, 768])
# # blocks.11.attn.proj.bias torch.Size([768])
# # blocks.11.norm2.weight torch.Size([768])
# # blocks.11.norm2.bias torch.Size([768])
# # blocks.11.mlp.lin1.weight torch.Size([3072, 768])
# # blocks.11.mlp.lin1.bias torch.Size([3072])
# # blocks.11.mlp.lin2.weight torch.Size([768, 3072])
# # blocks.11.mlp.lin2.bias torch.Size([768])
# # neck.0.weight torch.Size([256, 768, 1, 1])
# # neck.1.weight torch.Size([256])
# # neck.1.bias torch.Size([256])
# # neck.2.weight torch.Size([256, 256, 3, 3])
# # neck.3.weight torch.Size([256])
# # neck.3.bias torch.Size([256])

# # image_encoder.neck.0.weight
# # image_encoder.neck.1.weight
# # image_encoder.neck.1.bias
# # image_encoder.neck.2.weight
# # image_encoder.neck.3.weight
# # image_encoder.neck.3.bias
# # image_encoder.patch_embed.proj.weight
# # image_encoder.patch_embed.proj.bias
# # image_encoder.blocks.0.norm1.weight
# # image_encoder.blocks.0.norm1.bias
# # image_encoder.blocks.0.attn.rel_pos_h
# # image_encoder.blocks.0.attn.rel_pos_w
# # image_encoder.blocks.0.attn.qkv.weight
# # image_encoder.blocks.0.attn.qkv.bias
# # image_encoder.blocks.0.attn.proj.weight
# # image_encoder.blocks.0.attn.proj.bias
# # image_encoder.blocks.0.norm2.weight
# # image_encoder.blocks.0.norm2.bias
# # image_encoder.blocks.0.mlp.lin1.weight
# # image_encoder.blocks.0.mlp.lin1.bias
# # image_encoder.blocks.0.mlp.lin2.weight
# # image_encoder.blocks.0.mlp.lin2.bias
# # image_encoder.blocks.1.norm1.weight
# # image_encoder.blocks.1.norm1.bias
# # image_encoder.blocks.1.attn.rel_pos_h
# # image_encoder.blocks.1.attn.rel_pos_w
# # image_encoder.blocks.1.attn.qkv.weight
# # image_encoder.blocks.1.attn.qkv.bias
# # image_encoder.blocks.1.attn.proj.weight
# # image_encoder.blocks.1.attn.proj.bias
# # image_encoder.blocks.1.norm2.weight
# # image_encoder.blocks.1.norm2.bias
# # image_encoder.blocks.1.mlp.lin1.weight
# # image_encoder.blocks.1.mlp.lin1.bias
# # image_encoder.blocks.1.mlp.lin2.weight
# # image_encoder.blocks.1.mlp.lin2.bias
# # image_encoder.blocks.2.norm1.weight
# # image_encoder.blocks.2.norm1.bias
# # image_encoder.blocks.2.attn.rel_pos_h
# # image_encoder.blocks.2.attn.rel_pos_w
# # image_encoder.blocks.2.attn.qkv.weight
# # image_encoder.blocks.2.attn.qkv.bias
# # image_encoder.blocks.2.attn.proj.weight
# # image_encoder.blocks.2.attn.proj.bias
# # image_encoder.blocks.2.norm2.weight
# # image_encoder.blocks.2.norm2.bias
# # image_encoder.blocks.2.mlp.lin1.weight
# # image_encoder.blocks.2.mlp.lin1.bias
# # image_encoder.blocks.2.mlp.lin2.weight
# # image_encoder.blocks.2.mlp.lin2.bias
# # image_encoder.blocks.3.norm1.weight
# # image_encoder.blocks.3.norm1.bias
# # image_encoder.blocks.3.attn.rel_pos_h
# # image_encoder.blocks.3.attn.rel_pos_w
# # image_encoder.blocks.3.attn.qkv.weight
# # image_encoder.blocks.3.attn.qkv.bias
# # image_encoder.blocks.3.attn.proj.weight
# # image_encoder.blocks.3.attn.proj.bias
# # image_encoder.blocks.3.norm2.weight
# # image_encoder.blocks.3.norm2.bias
# # image_encoder.blocks.3.mlp.lin1.weight
# # image_encoder.blocks.3.mlp.lin1.bias
# # image_encoder.blocks.3.mlp.lin2.weight
# # image_encoder.blocks.3.mlp.lin2.bias
# # image_encoder.blocks.4.norm1.weight
# # image_encoder.blocks.4.norm1.bias
# # image_encoder.blocks.4.attn.rel_pos_h
# # image_encoder.blocks.4.attn.rel_pos_w
# # image_encoder.blocks.4.attn.qkv.weight
# # image_encoder.blocks.4.attn.qkv.bias
# # image_encoder.blocks.4.attn.proj.weight
# # image_encoder.blocks.4.attn.proj.bias
# # image_encoder.blocks.4.norm2.weight
# # image_encoder.blocks.4.norm2.bias
# # image_encoder.blocks.4.mlp.lin1.weight
# # image_encoder.blocks.4.mlp.lin1.bias
# # image_encoder.blocks.4.mlp.lin2.weight
# # image_encoder.blocks.4.mlp.lin2.bias
# # image_encoder.blocks.5.norm1.weight
# # image_encoder.blocks.5.norm1.bias
# # image_encoder.blocks.5.attn.rel_pos_h
# # image_encoder.blocks.5.attn.rel_pos_w
# # image_encoder.blocks.5.attn.qkv.weight
# # image_encoder.blocks.5.attn.qkv.bias
# # image_encoder.blocks.5.attn.proj.weight
# # image_encoder.blocks.5.attn.proj.bias
# # image_encoder.blocks.5.norm2.weight
# # image_encoder.blocks.5.norm2.bias
# # image_encoder.blocks.5.mlp.lin1.weight
# # image_encoder.blocks.5.mlp.lin1.bias
# # image_encoder.blocks.5.mlp.lin2.weight
# # image_encoder.blocks.5.mlp.lin2.bias
# # image_encoder.blocks.6.norm1.weight
# # image_encoder.blocks.6.norm1.bias
# # image_encoder.blocks.6.attn.rel_pos_h
# # image_encoder.blocks.6.attn.rel_pos_w
# # image_encoder.blocks.6.attn.qkv.weight
# # image_encoder.blocks.6.attn.qkv.bias
# # image_encoder.blocks.6.attn.proj.weight
# # image_encoder.blocks.6.attn.proj.bias
# # image_encoder.blocks.6.norm2.weight
# # image_encoder.blocks.6.norm2.bias
# # image_encoder.blocks.6.mlp.lin1.weight
# # image_encoder.blocks.6.mlp.lin1.bias
# # image_encoder.blocks.6.mlp.lin2.weight
# # image_encoder.blocks.6.mlp.lin2.bias
# # image_encoder.blocks.7.norm1.weight
# # image_encoder.blocks.7.norm1.bias
# # image_encoder.blocks.7.attn.rel_pos_h
# # image_encoder.blocks.7.attn.rel_pos_w
# # image_encoder.blocks.7.attn.qkv.weight
# # image_encoder.blocks.7.attn.qkv.bias
# # image_encoder.blocks.7.attn.proj.weight
# # image_encoder.blocks.7.attn.proj.bias
# # image_encoder.blocks.7.norm2.weight
# # image_encoder.blocks.7.norm2.bias
# # image_encoder.blocks.7.mlp.lin1.weight
# # image_encoder.blocks.7.mlp.lin1.bias
# # image_encoder.blocks.7.mlp.lin2.weight
# # image_encoder.blocks.7.mlp.lin2.bias
# # image_encoder.blocks.8.norm1.weight
# # image_encoder.blocks.8.norm1.bias
# # image_encoder.blocks.8.attn.rel_pos_h
# # image_encoder.blocks.8.attn.rel_pos_w
# # image_encoder.blocks.8.attn.qkv.weight
# # image_encoder.blocks.8.attn.qkv.bias
# # image_encoder.blocks.8.attn.proj.weight
# # image_encoder.blocks.8.attn.proj.bias
# # image_encoder.blocks.8.norm2.weight
# # image_encoder.blocks.8.norm2.bias
# # image_encoder.blocks.8.mlp.lin1.weight
# # image_encoder.blocks.8.mlp.lin1.bias
# # image_encoder.blocks.8.mlp.lin2.weight
# # image_encoder.blocks.8.mlp.lin2.bias
# # image_encoder.blocks.9.norm1.weight
# # image_encoder.blocks.9.norm1.bias
# # image_encoder.blocks.9.attn.rel_pos_h
# # image_encoder.blocks.9.attn.rel_pos_w
# # image_encoder.blocks.9.attn.qkv.weight
# # image_encoder.blocks.9.attn.qkv.bias
# # image_encoder.blocks.9.attn.proj.weight
# # image_encoder.blocks.9.attn.proj.bias
# # image_encoder.blocks.9.norm2.weight
# # image_encoder.blocks.9.norm2.bias
# # image_encoder.blocks.9.mlp.lin1.weight
# # image_encoder.blocks.9.mlp.lin1.bias
# # image_encoder.blocks.9.mlp.lin2.weight
# # image_encoder.blocks.9.mlp.lin2.bias
# # image_encoder.blocks.10.norm1.weight
# # image_encoder.blocks.10.norm1.bias
# # image_encoder.blocks.10.attn.rel_pos_h
# # image_encoder.blocks.10.attn.rel_pos_w
# # image_encoder.blocks.10.attn.qkv.weight
# # image_encoder.blocks.10.attn.qkv.bias
# # image_encoder.blocks.10.attn.proj.weight
# # image_encoder.blocks.10.attn.proj.bias
# # image_encoder.blocks.10.norm2.weight
# # image_encoder.blocks.10.norm2.bias
# # image_encoder.blocks.10.mlp.lin1.weight
# # image_encoder.blocks.10.mlp.lin1.bias
# # image_encoder.blocks.10.mlp.lin2.weight
# # image_encoder.blocks.10.mlp.lin2.bias
# # image_encoder.blocks.11.norm1.weight
# # image_encoder.blocks.11.norm1.bias
# # image_encoder.blocks.11.attn.rel_pos_h
# # image_encoder.blocks.11.attn.rel_pos_w
# # image_encoder.blocks.11.attn.qkv.weight
# # image_encoder.blocks.11.attn.qkv.bias
# # image_encoder.blocks.11.attn.proj.weight
# # image_encoder.blocks.11.attn.proj.bias
# # image_encoder.blocks.11.norm2.weight
# # image_encoder.blocks.11.norm2.bias
# # image_encoder.blocks.11.mlp.lin1.weight
# # image_encoder.blocks.11.mlp.lin1.bias
# # image_encoder.blocks.11.mlp.lin2.weight
# # image_encoder.blocks.11.mlp.lin2.bias