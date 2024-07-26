"""This file contains the configuration settings for the market environment."""

from environ.settings import PROJECT_ROOT

# Paths
DATA_PATH = PROJECT_ROOT / "data"
FIGURE_PATH = PROJECT_ROOT / "figures"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data"

# Global constants
MAXLEN = 256
BATCH_SIZE = 32


SEEDS = [
    "法币",
    "大饼",
    "姨太",
    "柚子",
    "U",
    "辣条",
    "白皮书",
    "韭菜",
    # "项目方",
    # "链圈",
    # "矿圈",
    "狗庄",
    "巨鲸",
    "建仓",
    "锁仓",
    "新仓",
    "全仓",
    "空仓",
    "瀑布",
    "割肉",
    "腰斩",
    "梭哈",
    # "币本位",
    # "交易对",
    "合约",
    "空投",
    "糖果",
    "破发",
    "私募",
    "杠杆",
    "跳水",
    "踏空",
    "护盘",
    "牛市",
    "熊市",
    "挖矿",
    "矿工",
    "AMM",
    # "流动性挖矿",
    "农民",
    "科学家",
    # "无常损失",
    "土狗",
]