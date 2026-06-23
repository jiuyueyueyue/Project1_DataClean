"""
项目全局配置模块

集中管理所有可配置常量，消除代码内硬编码。
配置加载优先级：.env 环境变量 > 类内默认值

Usage:
    from config import settings
    raw_path = settings.RAW_DATA_PATH
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ============================================================================
# 项目根目录
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _load_env_file(env_path: Path) -> None:
    """从 .env 文件加载环境变量（简易实现，不依赖 python-dotenv）"""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


# 加载项目级 .env
_load_env_file(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    """读取环境变量，不存在时返回默认值"""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量"""
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量"""
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


# ============================================================================
# 配置数据类
# ============================================================================
@dataclass(frozen=True)
class Settings:
    """项目全局配置（不可变，线程安全）"""

    # ========================================================================
    # 路径配置 —— 所有文件路径集中在此
    # ========================================================================
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    RAW_DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    PROCESSED_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "output")

    RAW_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("RAW_DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "raw_data.csv"))
    ))
    CLEAN_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("CLEAN_DATA_PATH", str(PROJECT_ROOT / "data" / "processed" / "clean_data.csv"))
    ))
    LABELED_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("LABELED_DATA_PATH", str(PROJECT_ROOT / "data" / "labeled_data.csv"))
    ))
    LABEL_PLOT_PATH: Path = field(default_factory=lambda: Path(
        _env("LABEL_PLOT_PATH", str(PROJECT_ROOT / "data" / "output" / "label_distribution.png"))
    ))
    LOG_FILE_PATH: Path = field(default_factory=lambda: Path(
        _env("LOG_FILE_PATH", str(PROJECT_ROOT / "data" / "output" / "pipeline.log"))
    ))

    # ========================================================================
    # 数据生成配置
    # ========================================================================
    DEVICE_LIST: List[str] = field(default_factory=lambda: [
        "空调", "客厅灯光", "电动窗帘", "监控摄像头", "扫地机器人"
    ])
    QUESTION_LIST: List[str] = field(default_factory=lambda: [
        "怎么调节温度", "开机报错怎么办", "如何连接WiFi", "定时功能怎么设置", "离线无法控制"
    ])
    DATA_SOURCE_LABEL: str = "家电论坛爬虫采集"
    GENERATE_TOTAL_COUNT: int = _env_int("GENERATE_TOTAL_COUNT", 10000)

    # ========================================================================
    # 数据清洗配置
    # ========================================================================
    TEXT_MIN_LENGTH: int = _env_int("TEXT_MIN_LENGTH", 5)
    CLEAN_TARGET_COUNT: int = _env_int("CLEAN_TARGET_COUNT", 5800)
    # CSV 读取编码
    CSV_ENCODING: str = _env("CSV_ENCODING", "utf-8")

    # ========================================================================
    # 数据分析 / 可视化配置
    # ========================================================================
    PLOT_FIGSIZE: Tuple[int, int] = (12, 6)
    PLOT_DPI: int = _env_int("PLOT_DPI", 300)
    PLOT_COLOR: str = _env("PLOT_COLOR", "#4285F4")
    PLOT_TITLE: str = "智能家居问答数据集标签分布统计"
    PLOT_YLABEL: str = "样本数量"

    # ========================================================================
    # 模型训练配置
    # ========================================================================
    # TF-IDF
    TFIDF_MAX_FEATURES: int = _env_int("TFIDF_MAX_FEATURES", 5000)
    # 数据集划分
    TRAIN_TEST_SPLIT_RATIO: float = _env_float("TRAIN_TEST_SPLIT_RATIO", 0.2)
    RANDOM_SEED: int = _env_int("RANDOM_SEED", 42)
    # 随机森林
    RF_N_ESTIMATORS: int = _env_int("RF_N_ESTIMATORS", 100)

    # ========================================================================
    # 日志配置
    # ========================================================================
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # ========================================================================
    # 运行时配置
    # ========================================================================
    # 是否在数据分析完成后显示图表窗口
    SHOW_PLOT: bool = _env("SHOW_PLOT", "true").lower() == "true"

    def __post_init__(self) -> None:
        """初始化后自动创建必要目录"""
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
