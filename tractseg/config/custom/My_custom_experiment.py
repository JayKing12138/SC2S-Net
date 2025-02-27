from tractseg.experiments.base import Config as BaseConfig

class Config(BaseConfig):
    # 自定义参数
    EXP_NAME = "my_custom_experiment"
    LR = 0.001
    BATCH_SIZE = 4
    EPOCHS = 200
    # 其他参数根据需要设置
