"""
工具模块包初始化

提供项目内通用的工具函数：
- logger:    统一日志体系（控制台 + 文件双通道）
- io_utils:  文件安全读写（异常捕获 + 目录自建）
- label_parser: Label Studio 标注数据安全解析（ast.literal_eval 替代 eval）
"""

from utils.logger import setup_logger
from utils.io_utils import safe_read_csv, safe_write_csv
from utils.label_parser import parse_label_studio_annotations

__all__ = [
    "setup_logger",
    "safe_read_csv",
    "safe_write_csv",
    "parse_label_studio_annotations",
]
