"""
电商智能客服知识库 & 对话语料预处理 —— 项目全局配置模块

集中管理所有可配置常量，消除代码内硬编码。
配置加载优先级：.env 环境变量 > 类内默认值

业务场景: 对齐京小智、飞鸽后台数据生产流程，
         为意图识别模型和 RAG 向量库提供标准化数据集输入。

Usage:
    from config import settings
    faq_categories = settings.FAQ_CATEGORIES
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


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
    """电商智能客服项目全局配置（不可变，线程安全）"""

    # ========================================================================
    # 路径配置
    # ========================================================================
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    RAW_DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    PROCESSED_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "output")

    # 四类业务数据输出路径
    FAQ_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("FAQ_DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "faq_data.csv"))
    ))
    PRODUCT_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("PRODUCT_DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "product_data.csv"))
    ))
    AFTERSALES_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("AFTERSALES_DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "aftersales_data.csv"))
    ))
    CHATLOG_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("CHATLOG_DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "chatlog_data.csv"))
    ))

    # 清洗后统一输出路径
    CLEAN_DATA_PATH: Path = field(default_factory=lambda: Path(
        _env("CLEAN_DATA_PATH", str(PROJECT_ROOT / "data" / "processed" / "clean_corpus.csv"))
    ))

    # RAG 切片导出路径
    KB_CHUNKS_PATH: Path = field(default_factory=lambda: Path(
        _env("KB_CHUNKS_PATH", str(PROJECT_ROOT / "data" / "output" / "kb_chunks.jsonl"))
    ))

    # 可视化 & 日志
    CORPUS_PLOT_PATH: Path = field(default_factory=lambda: Path(
        _env("CORPUS_PLOT_PATH", str(PROJECT_ROOT / "data" / "output" / "corpus_distribution.png"))
    ))
    LOG_FILE_PATH: Path = field(default_factory=lambda: Path(
        _env("LOG_FILE_PATH", str(PROJECT_ROOT / "data" / "output" / "pipeline.log"))
    ))

    # ========================================================================
    # FAQ 问答对数据配置
    # ========================================================================
    FAQ_CATEGORIES: Dict[str, List[str]] = field(default_factory=lambda: {
        "订单查询": ["如何查询订单物流状态", "订单显示已发货但没收到", "如何修改订单地址",
                     "怎么查看历史订单", "订单取消后多久退款", "如何申请电子发票"],
        "退换货": ["退货流程是什么", "换货需要多长时间", "退货运费谁承担",
                   "超过7天还能退货吗", "换货流程怎么操作", "退货后优惠券会退回吗"],
        "物流配送": ["预计多久送达", "可以指定配送时间吗", "快递没送到显示已签收",
                     "能修改收货地址吗", "境外可以配送吗", "到货后如何验货"],
        "会员权益": ["会员等级怎么升级", "积分如何获取和使用", "会员有什么专属优惠",
                     "生日礼包在哪里领取", "积分会过期吗", "会员等级会降级吗"],
        "支付问题": ["支持哪些支付方式", "支付失败怎么办", "可以分期付款吗",
                     "如何绑定银行卡", "支付时提示限额怎么办", "货到付款支持吗"],
        "账户安全": ["如何修改密码", "手机号换绑流程", "账户被盗怎么办",
                     "如何注销账户", "实名认证需要什么", "异地登录提醒怎么设置"],
        "商品咨询": ["这款商品有优惠吗", "尺码怎么选", "支持哪些颜色",
                     "质保期多久", "有现货吗", "可以定制吗"],
        "活动优惠": ["优惠券怎么领取", "活动什么时候开始", "满减规则是什么",
                     "秒杀商品能抢到吗", "优惠券可以叠加吗", "预售商品怎么付尾款"],
    })
    FAQ_ANSWER_TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "订单查询": "您可以在「我的订单」中找到对应订单，点击「查看详情」即可获取{detail}。如有疑问可随时联系在线客服。",
        "退换货": "您好，{detail}。请在「我的订单」中选择对应订单提交{action}申请，客服将在24小时内审核处理。",
        "物流配送": "您的订单已由{carrier}承运，运单号 {tracking}。{detail}，预计{eta}送达。可在「物流详情」中实时查看。",
        "会员权益": "感谢您对会员体系的关注！{detail}。您可在「会员中心」查看完整权益说明和当前等级进度。",
        "支付问题": "关于支付问题，{detail}。如仍无法解决，建议尝试切换支付方式或联系发卡行确认卡片状态。",
        "账户安全": "账户安全是我们的首要任务。{detail}。您可在「账户设置-安全中心」中完成相关操作。",
        "商品咨询": "感谢您的关注！{detail}。您可在商品详情页查看完整规格参数和用户评价。",
        "活动优惠": "关于活动优惠，{detail}。建议关注首页活动入口或开启促销提醒，不错过优惠信息。",
    })
    FAQ_GENERATE_COUNT: int = _env_int("FAQ_GENERATE_COUNT", 3000)

    # ========================================================================
    # 商品资料数据配置
    # ========================================================================
    PRODUCT_CATEGORIES: Dict[str, List[str]] = field(default_factory=lambda: {
        "手机通讯": ["智能手机", "功能手机", "手机配件"],
        "电脑办公": ["笔记本", "台式机", "平板电脑"],
        "家电": ["空调", "洗衣机", "冰箱", "扫地机器人"],
        "服饰": ["男装", "女装", "鞋靴", "箱包"],
        "家居": ["沙发", "床垫", "厨具", "灯具"],
        "美妆": ["护肤品", "彩妆", "香水", "个人护理"],
        "食品": ["零食", "饮料", "生鲜", "保健品"],
        "运动户外": ["跑步鞋", "健身器材", "露营装备", "骑行"],
    })
    PRODUCT_BRANDS: List[str] = field(default_factory=lambda: [
        "华为", "小米", "苹果", "三星", "海尔", "美的", "格力", "科沃斯",
        "耐克", "阿迪达斯", "安踏", "李宁", "欧莱雅", "兰蔻", "资生堂",
        "三只松鼠", "良品铺子", "百草味", "蒙牛", "伊利",
    ])
    PRODUCT_GENERATE_COUNT: int = _env_int("PRODUCT_GENERATE_COUNT", 2000)

    # ========================================================================
    # 售后规则数据配置
    # ========================================================================
    AFTERSALES_RULES: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: {
        "退换货政策": [
            {"title": "7天无理由退货", "content": "商品完好、未使用、不影响二次销售，自签收之日起7天内可申请无理由退货。退回运费由买家承担（质量问题除外）。退货商品需保持原包装完整，配件齐全。"},
            {"title": "15天质量问题换货", "content": "自签收之日起15天内，如商品出现非人为损坏的质量问题，可申请换货。经售后检测确认后，免费更换同型号商品。如无同型号，可与用户协商更换其他商品或退款。"},
            {"title": "大件商品退货", "content": "大家电、家具等大件商品退货需由指定物流上门取件。用户需保留原包装至少30天。上门取件费用由责任方承担：质量问题由商家承担，无理由退货由买家承担。"},
            {"title": "跨境商品退货", "content": "跨境商品支持7天无理由退货，但需确保商品未拆封、防伪标识完好。退回至国内退货仓，不支持直接退回境外。跨境商品不支持换货，仅支持退货退款。"},
        ],
        "保修维修": [
            {"title": "全国联保政策", "content": "品牌商品享受全国联保服务。保修期内凭购买凭证可在任意授权维修点享受免费维修。保修期自购买之日起计算，不同品类保修期限以商品详情页标注为准。"},
            {"title": "延保服务说明", "content": "用户可在购买商品时加购延保服务，延保期限可选1-3年。延保期间享受与正常保修相同的服务标准。延保不支持单独退订，随主商品退货时一并退款。"},
        ],
        "发票相关": [
            {"title": "电子发票开具", "content": "订单完成后系统自动开具电子发票，可在「我的订单-发票详情」中查看和下载。支持个人和公司抬头，公司抬头的发票需填写完整税号信息。"},
            {"title": "发票修改与补开", "content": "发票开具后如需修改抬头信息，可在订单完成后30天内在线申请修改。超过时效需联系人工客服处理。纸质发票可申请补开，收取快递费用。"},
        ],
        "物流赔付": [
            {"title": "物流损坏赔付", "content": "收到商品时发现外包装明显破损，请当场拍照留存并拒收。已签收后发现商品损坏，需在24小时内联系客服并提供开箱视频/照片。经核实属于物流责任的，由平台先行赔付。"},
            {"title": "物流超时赔付", "content": "延迟送达超过承诺时效，可获得相应补偿。具体补偿标准：延迟1-3天补偿5元优惠券，延迟3-7天补偿10元优惠券，延迟7天以上按订单金额1%赔付。"},
        ],
        "投诉建议": [
            {"title": "服务投诉处理", "content": "如对客服服务不满意，可通过「我的-投诉建议」提交投诉。投诉将在24小时内由专人跟进处理，72小时内给出处理结果。投诉内容请尽量提供截图等佐证材料以便快速处理。"},
        ],
    })
    AFTERSALES_GENERATE_COUNT: int = _env_int("AFTERSALES_GENERATE_COUNT", 500)

    # ========================================================================
    # 客服对话日志数据配置
    # ========================================================================
    CHATLOG_SCENARIOS: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "售前咨询": {
            "user_messages": [
                "这款商品有优惠吗", "和XX品牌比哪个好", "可以发一下详细参数吗",
                "现在下单什么时候能到", "支持花呗分期吗", "有赠品吗",
                "我想了解一下这个产品的功能", "有没有使用教程", "好评率多少",
                "这款适合送人吗", "和上一代有什么区别", "能再便宜点吗",
            ],
            "intent_labels": ["商品咨询", "比价", "促销询问", "功能咨询"],
        },
        "售中物流": {
            "user_messages": [
                "我的快递到哪了", "为什么物流信息一直不更新", "可以改地址吗",
                "明天能送到吗", "快递员电话多少", "我选了工作日配送怎么还没到",
                "物流显示签收但我没收到", "包裹外包装有破损", "可以指定周末配送吗",
                "快递放快递柜了我拿不到",
            ],
            "intent_labels": ["物流查询", "配送时效", "地址修改", "异常件处理"],
        },
        "售后投诉": {
            "user_messages": [
                "收到的商品有划痕", "颜色和图片不一样", "用了一个月就坏了",
                "退货退款进度怎么这么慢", "卖家不给退款怎么办", "我要投诉商家",
                "配件少发了", "说明书是坏的怎么看", "功能不能正常使用",
                "这是假货吧",
            ],
            "intent_labels": ["质量问题", "退换货", "投诉商家", "商品不符"],
        },
        "账户问题": {
            "user_messages": [
                "密码忘了怎么找回", "手机号换了怎么绑定新号", "怎么注销账户",
                "实名认证总是失败", "为什么我的账户被限制下单", "优惠券怎么不见了",
                "我的积分怎么变少了", "账号被异地登录了",
            ],
            "intent_labels": ["账户安全", "密码找回", "实名认证", "优惠券查询"],
        },
        "活动咨询": {
            "user_messages": [
                "618活动什么时候开始", "满300减50可以用吗", "秒杀几点开始",
                "优惠券为什么用不了", "预售商品什么时候发货", "能价保吗",
                "活动期间买的降价了能退差价吗", "怎么领这个活动的优惠券",
            ],
            "intent_labels": ["活动规则", "优惠券使用", "秒杀", "价格保护"],
        },
    })
    CHATLOG_GENERATE_COUNT: int = _env_int("CHATLOG_GENERATE_COUNT", 4000)

    # 客服话术模板（用于 agent 回复生成）
    AGENT_TEMPLATES: List[str] = field(default_factory=lambda: [
        "亲，感谢您的耐心等待~ 关于您提到的{issue}，我来帮您处理。",
        "您好，非常抱歉给您带来不便。{solution}，您看这样可以吗？",
        "感谢您的反馈！我已经帮您查询了，{detail}。请问还有其他可以帮您的吗？",
        "亲，您的问题我们已经记录下来了。{action}，预计{time}内给您答复。",
        "好的，我已了解您的情况。根据{policy}，{solution}。",
        "不客气呢，很高兴能帮到您！如后续还有疑问，随时联系我们哦~",
    ])

    # ========================================================================
    # 数据清洗配置
    # ========================================================================
    TEXT_MIN_LENGTH: int = _env_int("TEXT_MIN_LENGTH", 5)
    # 各数据类型清洗配置
    CLEAN_BY_TYPE: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "faq":        {"min_question_len": 4, "min_answer_len": 10},
        "product":    {"min_name_len": 2, "min_desc_len": 20},
        "aftersales": {"min_title_len": 2, "min_content_len": 30},
        "chatlog":    {"min_message_len": 2, "max_message_len": 500},
    })
    CSV_ENCODING: str = _env("CSV_ENCODING", "utf-8")

    # ========================================================================
    # 知识库切片预处理配置
    # ========================================================================
    KB_CHUNK_SIZE: int = _env_int("KB_CHUNK_SIZE", 512)      # 每个切片最大字符数
    KB_CHUNK_OVERLAP: int = _env_int("KB_CHUNK_OVERLAP", 128)  # 相邻切片重叠字符数
    KB_MIN_CHUNK_LENGTH: int = _env_int("KB_MIN_CHUNK_LENGTH", 50)  # 丢弃低于此长度的碎片
    KB_EXPORT_FORMAT: str = _env("KB_EXPORT_FORMAT", "jsonl")  # jsonl | csv | parquet
    # 参与切片的数据类型（默认 FAQ + 商品资料 + 售后规则；对话日志可选）
    KB_SOURCE_TYPES: List[str] = field(default_factory=lambda: ["faq", "product", "aftersales"])

    # ========================================================================
    # 数据分析 / 可视化配置
    # ========================================================================
    PLOT_FIGSIZE: Tuple[int, int] = (14, 8)
    PLOT_DPI: int = _env_int("PLOT_DPI", 300)
    PLOT_COLOR_PALETTE: List[str] = field(default_factory=lambda: [
        "#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8E24AA", "#00ACC1", "#FF7043", "#9E9D24"
    ])

    # ========================================================================
    # 模型训练配置
    # ========================================================================
    TFIDF_MAX_FEATURES: int = _env_int("TFIDF_MAX_FEATURES", 5000)
    TRAIN_TEST_SPLIT_RATIO: float = _env_float("TRAIN_TEST_SPLIT_RATIO", 0.2)
    RANDOM_SEED: int = _env_int("RANDOM_SEED", 42)
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
    SHOW_PLOT: bool = _env("SHOW_PLOT", "true").lower() == "true"

    def __post_init__(self) -> None:
        """初始化后自动创建必要目录"""
        for dir_path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
