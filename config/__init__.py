"""
配置模块包初始化

导出项目全局配置单例，供各业务模块通过 `from config import settings` 统一引用。
"""

from config.settings import Settings

# 全局配置单例 —— 项目内所有模块通过此实例获取配置
settings = Settings()

__all__ = ["settings"]
