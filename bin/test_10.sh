#!/bin/bash

# 设置最大值和日志文件路径
max_value=198
log_file="/home/crq/TractSeg/hcp_exp/HCP_TEST_x17/test_output_10.log"

# 初始化 i 为序列的起始值
i=0

# 使用 while 循环来迭代序列中的值
while [ $i -le $max_value ]; do
    # 计算当前的权重路径
    WEIGHTS_PATH="/home/crq/TractSeg/hcp_exp/HCP_TEST_x17/best_weights_ep${i}.npz"
    
    # 运行 ExpRunner.py 并将输出追加到日志文件中
    # 假设 ExpRunner.py 接受 --weights_path 参数
    python /home/crq/TractSeg/bin/ExpRunner.py --weights_path "$WEIGHTS_PATH" >> "$log_file"
    
    # 由于序列是每隔3个数字，所以 i 需要增加3
    i=$((i + 10))
    
    # 检查 i 是否超出了最大值
    if [ $i -gt $max_value ]; then
        break
    fi
done


# #!/bin/bash

# # 循环200次
# for i in {0..199}; do
#     # 替换 WEIGHTS_PATH 的值
#     WEIGHTS_PATH="/home/crq/TractSeg/hcp_exp/HCP_TEST_x17/best_weights_ep${i}.npz"
    
#     # 运行 ExpRunner.py 并传递 WEIGHTS_PATH 作为参数
#     # 假设 ExpRunner.py 接受一个参数作为权重路径
#     python /home/crq/TractSeg/bin/ExpRunner.py --weights_path "$WEIGHTS_PATH"
    
#     # 如果 ExpRunner.py 不需要任何参数，可以直接运行它
#     # python /home/crq/TractSeg/bin/ExpRunner.py
    
#     # 等待一段时间，或者进行一些其他操作
#     # sleep 1 # 如果需要等待一秒
# done