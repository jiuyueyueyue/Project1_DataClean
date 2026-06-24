"""
电商智能客服数据生成模块

生成四类标准化业务数据，对齐京小智、飞鸽后台数据生产流程：
  1. FAQ 问答对 —— 知识库核心语料，用于 RAG 检索
  2. 商品资料 —— 商品规格/描述，用于商品问答匹配
  3. 售后规则 —— 退换货/保修/发票政策，用于规则引擎和 RAG
  4. 客服对话日志 —— 多轮对话，用于意图识别模型训练

每类数据独立生成函数，统一通过 generate_all_data() 编排调用。

原文件对应关系:
  改造前: 基于 DEVICE_LIST × QUESTION_LIST 笛卡尔积的合成数据生成
  改造后: 四类业务数据独立生成器，模板驱动 + 随机采样
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from utils.io_utils import safe_write_csv

# ============================================================================
# 常量
# ============================================================================
_COURIERS: List[str] = ["顺丰速运", "京东物流", "中通快递", "圆通速递", "韵达快递", "EMS"]
_AGENT_NAMES: List[str] = ["小智", "小慧", "云客服-乐乐", "智能助手-飞鸽", "客服专员-小美"]


def _random_sample(data: List[Any], k: int) -> List[Any]:
    """从列表中随机采样 k 个元素（k 可超过列表长度，自动循环填充）"""
    if k <= len(data):
        return random.sample(data, k)
    # 超出时循环采样
    result: List[Any] = []
    while len(result) < k:
        result.extend(random.sample(data, min(len(data), k - len(result))))
    return result


# ============================================================================
# 1. FAQ 问答对生成
# ============================================================================
def generate_faq_data(
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    生成 FAQ 问答对数据

    基于 FAQ_CATEGORIES 中的品类-问题模板和 FAQ_ANSWER_TEMPLATES 中的回答模板，
    通过模板填充生成标准化问答对。每行包含:
      - type: "faq"
      - category: 品类名称
      - question: 用户问句
      - answer: 客服回答

    Args:
        output_path: 输出 CSV 路径，默认 settings.FAQ_DATA_PATH
        logger:      logger 实例

    Returns:
        FAQ 数据 DataFrame，列: [type, category, question, answer]
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.FAQ_DATA_PATH

    log.info(">>> 生成 FAQ 问答对数据...")
    rows: List[List[str]] = []

    for category, questions in settings.FAQ_CATEGORIES.items():
        template: str = settings.FAQ_ANSWER_TEMPLATES.get(category, "关于{category}的问题，建议您联系在线客服获取详细解答。")

        for question in questions:
            # 根据品类动态填充回答模板变量
            answer_kwargs: Dict[str, str] = {"category": category}
            if category == "订单查询":
                answer_kwargs["detail"] = "订单编号、物流状态和预计送达时间"
            elif category == "退换货":
                answer_kwargs["detail"] = "我们支持7天无理由退换货，商品需完好未使用且保留原包装"
                answer_kwargs["action"] = "退换货"
            elif category == "物流配送":
                courier = random.choice(_COURIERS)
                answer_kwargs["carrier"] = courier
                answer_kwargs["tracking"] = f"SF{random.randint(100000000, 999999999):d}"
                answer_kwargs["detail"] = f"当前快件已从{random.choice(['北京', '上海', '广州', '深圳', '杭州'])}分拨中心发出"
                answer_kwargs["eta"] = f"{random.randint(1,5)}个工作日"
            elif category == "会员权益":
                answer_kwargs["detail"] = "会员等级分为银卡/金卡/钻石卡，享有积分加速、专属折扣和生日礼包"
            elif category == "支付问题":
                answer_kwargs["detail"] = "建议您先检查银行卡/支付宝/微信余额是否充足，并确认是否为本人操作"
            elif category == "账户安全":
                answer_kwargs["detail"] = "建议您开启两步验证，定期修改密码，勿将验证码告知他人"
            elif category == "商品咨询":
                answer_kwargs["detail"] = "建议您查看商品详情页的规格参数、用户评价和使用说明"
            elif category == "活动优惠":
                answer_kwargs["detail"] = "建议关注首页活动入口或开启促销提醒，优惠券可在「我的-优惠券」中查看"

            try:
                answer = template.format(**answer_kwargs)
            except KeyError:
                answer = template  # fallback: 返回原模板
            rows.append(["faq", category, question, answer])

    # 通过重复采样达到目标条数
    if len(rows) < settings.FAQ_GENERATE_COUNT:
        rows = _random_sample(rows, settings.FAQ_GENERATE_COUNT)

    df = pd.DataFrame(rows, columns=["type", "category", "question", "answer"])
    log.info("FAQ 数据: 生成 %d 条 (目标 %d)", len(df), settings.FAQ_GENERATE_COUNT)
    safe_write_csv(df, output_path, logger=log)
    return df


# ============================================================================
# 2. 商品资料生成
# ============================================================================
def generate_product_data(
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    生成商品资料数据

    基于 PRODUCT_CATEGORIES 中的品类-子类映射和 PRODUCT_BRANDS 品牌库，
    通过模板拼接生成模拟商品记录。每行包含:
      - type: "product"
      - spu_id, spu_name, category, sub_category, brand, specs, description

    Args:
        output_path: 输出 CSV 路径，默认 settings.PRODUCT_DATA_PATH
        logger:      logger 实例

    Returns:
        商品数据 DataFrame
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.PRODUCT_DATA_PATH

    log.info(">>> 生成商品资料数据...")
    rows: List[List[str]] = []
    spu_counter = 0

    desc_templates: List[str] = [
        "{brand}{name}，{specs}。{category}品类热销产品，累计好评{rating}万+。{feature}",
        "{brand}官方正品{name}，{specs}。支持全国联保，{feature}。已通过{category}行业标准认证。",
        "【{brand}旗舰店】{name}，{specs}。{feature}。现在下单享{benefit}。",
    ]

    # 为每个品类-子类组合生成商品
    for category, sub_categories in settings.PRODUCT_CATEGORIES.items():
        for sub_cat in sub_categories:
            # 每个子类生成一定数量的商品
            n_products = max(1, settings.PRODUCT_GENERATE_COUNT // len(sub_categories) // len(settings.PRODUCT_CATEGORIES))
            for _ in range(n_products):
                spu_counter += 1
                spu_id = f"SPU{spu_counter:06d}"
                brand = random.choice(settings.PRODUCT_BRANDS)
                name = f"{brand} {sub_cat} {spu_id[-3:]}型"
                specs_options = [
                    f"规格: {random.choice(['标准版', '升级版', '旗舰版', 'Pro版', 'Lite版'])}",
                    f"颜色: {random.choice(['经典黑', '珍珠白', '星空灰', '雾霾蓝', '樱花粉'])}",
                    f"重量: {random.uniform(0.1, 25.0):.1f}kg",
                ]
                specs = " | ".join(random.sample(specs_options, k=random.randint(1, 3)))
                rating = random.randint(1, 50)
                feature_options: Dict[str, List[str]] = {
                    "手机通讯": ["支持5G全网通", "高清AI四摄", "超长续航5000mAh", "120Hz高刷屏"],
                    "电脑办公": ["轻薄便携仅1.2kg", "14英寸2K屏", "i7处理器/16G内存", "12小时续航"],
                    "家电": ["一级能效", "智能APP远程控制", "静音运行低至22dB", "变频节能"],
                    "服饰": ["透气吸汗面料", "立体剪裁修身版型", "不起球不褪色", "可机洗免熨烫"],
                    "家居": ["实木环保材质", "人体工学设计", "多功能收纳", "北欧简约风格"],
                    "美妆": ["温和不刺激", "持妆12小时", "敏感肌适用", "补水保湿"],
                    "食品": ["精选优质原料", "独立小包装", "0添加防腐剂", "冷链配送"],
                    "运动户外": ["防滑耐磨大底", "透气网面鞋面", "轻量化设计", "专业级防护"],
                }
                feature = random.choice(feature_options.get(category, ["品质保证，值得信赖"]))
                benefit_options = ["限时包邮", "买二免一", "赠品随单送", "30天价保"]
                description = random.choice(desc_templates).format(
                    brand=brand, name=name, specs=specs, category=category,
                    rating=rating, feature=feature,
                    benefit=random.choice(benefit_options),
                )
                rows.append(["product", spu_id, name, category, sub_cat, brand, specs, description])

    df = pd.DataFrame(rows, columns=["type", "spu_id", "spu_name", "category", "sub_category", "brand", "specs", "description"])
    # 截断到目标量
    if len(df) > settings.PRODUCT_GENERATE_COUNT:
        df = df.sample(n=settings.PRODUCT_GENERATE_COUNT, random_state=settings.RANDOM_SEED).reset_index(drop=True)
    log.info("商品数据: 生成 %d 条 (目标 %d)", len(df), settings.PRODUCT_GENERATE_COUNT)
    safe_write_csv(df, output_path, logger=log)
    return df


# ============================================================================
# 3. 售后规则生成
# ============================================================================
def generate_aftersales_data(
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    生成售后规则数据

    基于 AFTERSALES_RULES 中的政策分类-规则条目生成标准化规则文档。
    每行包含:
      - type: "aftersales"
      - rule_id, rule_category, rule_title, rule_content

    Args:
        output_path: 输出 CSV 路径，默认 settings.AFTERSALES_DATA_PATH
        logger:      logger 实例

    Returns:
        售后规则数据 DataFrame
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.AFTERSALES_DATA_PATH

    log.info(">>> 生成售后规则数据...")
    rows: List[List[str]] = []
    rule_counter = 0

    for category, rules in settings.AFTERSALES_RULES.items():
        for rule in rules:
            rule_counter += 1
            rule_id = f"RULE{rule_counter:04d}"
            rows.append(["aftersales", rule_id, category, rule["title"], rule["content"]])

    # 通过重复填充达到目标量
    if len(rows) < settings.AFTERSALES_GENERATE_COUNT:
        rows = _random_sample(rows, settings.AFTERSALES_GENERATE_COUNT)

    df = pd.DataFrame(rows, columns=["type", "rule_id", "rule_category", "rule_title", "rule_content"])
    log.info("售后规则数据: 生成 %d 条 (目标 %d)", len(df), settings.AFTERSALES_GENERATE_COUNT)
    safe_write_csv(df, output_path, logger=log)
    return df


# ============================================================================
# 4. 客服对话日志生成
# ============================================================================
def generate_chatlog_data(
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    生成客服对话日志数据

    基于 CHATLOG_SCENARIOS 中的场景-话术模板生成模拟多轮对话。
    每行代表一轮对话:
      - type: "chatlog"
      - session_id: 会话ID
      - turn_id: 轮次序号
      - role: "user" | "agent"
      - message: 对话内容
      - intent_label: 意图标签（仅 user 消息）

    Args:
        output_path: 输出 CSV 路径，默认 settings.CHATLOG_DATA_PATH
        logger:      logger 实例

    Returns:
        对话日志 DataFrame
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.CHATLOG_DATA_PATH

    log.info(">>> 生成客服对话日志数据...")
    rows: List[List[str]] = []
    session_id = 0

    # 场景覆盖的意图标签
    scenario_intents: Dict[str, str] = {
        "售前咨询": "商品咨询",
        "售中物流": "物流查询",
        "售后投诉": "售后投诉",
        "账户问题": "账户管理",
        "活动咨询": "活动咨询",
    }

    total_turns = 0
    while total_turns < settings.CHATLOG_GENERATE_COUNT:
        session_id += 1
        sid = f"S{session_id:05d}"

        # 随机选择场景
        scenario_key = random.choice(list(settings.CHATLOG_SCENARIOS.keys()))
        scenario = settings.CHATLOG_SCENARIOS[scenario_key]

        # 随机对话轮次数 (2-5轮)
        n_turns = random.randint(2, 5)
        # 随机选择该场景下的 user 消息
        selected_user_msgs = _random_sample(scenario["user_messages"], n_turns)

        for turn_idx, user_msg in enumerate(selected_user_msgs, start=1):
            # User 消息
            intent = random.choice(scenario["intent_labels"])
            rows.append(["chatlog", sid, str(turn_idx), "user", user_msg, intent])

            # Agent 回复
            agent_template = random.choice(settings.AGENT_TEMPLATES)
            agent_msg = agent_template.format(
                issue=user_msg,
                solution="我已为您记录并转交相关部门处理",
                detail=f"关于「{user_msg[:20]}」的详细说明",
                action="我们会尽快核实情况",
                time=f"{random.randint(10, 60)}分钟",
                policy="平台售后政策",
            )
            rows.append(["chatlog", sid, str(turn_idx), "agent", agent_msg, ""])

        total_turns = len(rows)

    # 截断到目标量
    if len(rows) > settings.CHATLOG_GENERATE_COUNT:
        rows = rows[:settings.CHATLOG_GENERATE_COUNT]

    df = pd.DataFrame(rows, columns=["type", "session_id", "turn_id", "role", "message", "intent_label"])
    log.info("对话日志数据: 生成 %d 条 (目标 %d), %d 个会话", len(df), settings.CHATLOG_GENERATE_COUNT, session_id)
    safe_write_csv(df, output_path, logger=log)
    return df


# ============================================================================
# 5. 统一编排入口
# ============================================================================
def generate_all_data(
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """
    依次生成全部四类业务数据

    Args:
        logger: logger 实例

    Returns:
        字典: {"faq": DataFrame, "product": DataFrame,
               "aftersales": DataFrame, "chatlog": DataFrame}
    """
    log = logger or logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("  电商智能客服数据生成 —— 四类业务数据")
    log.info("=" * 60)

    results: Dict[str, pd.DataFrame] = {}

    # FAQ 问答对
    try:
        results["faq"] = generate_faq_data(logger=log)
    except Exception as e:
        log.exception("FAQ 数据生成失败: %s", e)
        results["faq"] = pd.DataFrame()

    # 商品资料
    try:
        results["product"] = generate_product_data(logger=log)
    except Exception as e:
        log.exception("商品数据生成失败: %s", e)
        results["product"] = pd.DataFrame()

    # 售后规则
    try:
        results["aftersales"] = generate_aftersales_data(logger=log)
    except Exception as e:
        log.exception("售后规则数据生成失败: %s", e)
        results["aftersales"] = pd.DataFrame()

    # 客服对话日志
    try:
        results["chatlog"] = generate_chatlog_data(logger=log)
    except Exception as e:
        log.exception("对话日志生成失败: %s", e)
        results["chatlog"] = pd.DataFrame()

    total = sum(len(df) for df in results.values())
    log.info("全部数据生成完成，总计 %d 条", total)
    return results


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "data_generator",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 电商数据生成模块")
    logger.info("=" * 60)

    try:
        result = generate_all_data(logger=logger)
        for data_type, df in result.items():
            if not df.empty:
                logger.info("[%s] 预览:\n%s", data_type, df.head(2).to_string())
    except Exception as e:
        logger.exception("数据生成失败: %s", e)
