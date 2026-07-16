"""
风格改写模块
============
使用检索到的八月长安风格段落作为 few-shot 示例，
调用智谱 GLM-4 模型将输入文本改写为八月长安风格。
"""

import os
from typing import Optional

from openai import OpenAI

from .retrieval import search


# ─── 智谱 API 配置 ───────────────────────────────────────────
ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4-flash"  # 免费模型，如需更好效果可改 "glm-4"


# ─── 多模型提供商配置 ───────────────────────────────────────────
# 所有提供商均兼容 OpenAI SDK，只需切换 base_url + api_key + model
MODEL_CONFIG = {
    # 智谱 GLM
    "glm-4-flash": {"base_url": "https://open.bigmodel.cn/api/paas/v4/", "provider": "智谱", "label": "GLM-4-Flash", "env_key": "ZHIPU_API_KEY"},
    "glm-4":       {"base_url": "https://open.bigmodel.cn/api/paas/v4/", "provider": "智谱", "label": "GLM-4",       "env_key": "ZHIPU_API_KEY"},
    # DeepSeek
    "deepseek-chat": {"base_url": "https://api.deepseek.com", "provider": "DeepSeek", "label": "DeepSeek-V3", "env_key": "DEEPSEEK_API_KEY"},
    # 通义千问
    "qwen-plus": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "provider": "阿里", "label": "Qwen-Plus", "env_key": "DASHSCOPE_API_KEY"},
    "qwen-max":  {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "provider": "阿里", "label": "Qwen-Max",  "env_key": "DASHSCOPE_API_KEY"},
    # 月之暗面 Kimi
    "moonshot-v1-8k": {"base_url": "https://api.moonshot.cn/v1", "provider": "月之暗面", "label": "Kimi", "env_key": "MOONSHOT_API_KEY"},
    # MiniMax
    "MiniMax-M1": {"base_url": "https://api.minimaxi.com/v1", "provider": "MiniMax", "label": "MiniMax-M1", "env_key": "MINIMAX_API_KEY"},
}


# ─── 八月长安风格特征说明 ──────────────────────────────────────
STYLE_DESCRIPTION = """\
八月长安的写作风格具有以下核心特征：

**一、意外具象——细节要"准"而非"美"**
不直接说"难过"、"开心"、"喜欢"，而是用一个出人意料的具象画面来承载情绪。
关键是：细节必须具体、意外、有生活质感，不能是读者一眼就能猜到的套路。
- ✅ "可乐罐里面的气泡争先恐后地破裂" —— 声音具体，画面鲜活，"争先恐后"带着拟人感
- ✅ "冰激凌似的天空层层渲染，让人分不清头顶到底是什么颜色" —— 比喻意外，"分不清"本身就是情绪
- ❌ "灰蒙蒙的天空" "冷风吹过" "咖啡早已冷却" —— 这些是万能忧伤模板，任何场景都能套，因此什么都没说

**二、情感间接性——不映射，要错位**
环境和情绪不是直接对应关系（难过→阴天，开心→晴天）。
八月长安的做法是：让角色在一个与情绪"不匹配"的场景里，因为某个微小的触发物突然崩塌。
情绪藏在缝隙里，不在表面上。
- 觉察到失望，不是写"天灰了"，而是写"太阳早已不知踪影，可天还没有黑"——那个"还没黑"的尴尬过渡期，就是失望本身
- 觉察到心动，不是写"心跳加速"，而是写"千里迢迢到达我耳边，他说，耿耿，你真有趣"——用距离感和声音细节代替生理反应

**三、对话有潜台词——说一套想一套**
角色说的话和心里想的不一样，对话表面平淡甚至嬉笑，底层藏着更深的情感。
- "老师，我没听懂" —— 表面是提问，实际是帮喜欢的人解围
- "叫我芊芊" —— 表面是改名玩笑，实际是在逃避某个名字带来的情绪

**四、克制与留白——情绪不写满**
最动人的时刻是欲言又止的沉默，是"算了"两个字背后的千言万语。
写一半，留一半，让读者自己去补完那个没说出口的句子。
- "声音断在晚风里" —— 不说"他走了我很难过"，让声音的消失代替告别
- "算了，好好考试吧" —— 所有的不甘和喜欢都压进这两个字里

**五、叙述者的声音——有温度的旁观**
成年叙述者回望少年时光，语气里带着自嘲、温柔和一点点心疼。
叙述者不是冷冰冰的摄像机，而是一个有态度的人在讲故事。
- "穿了这么多年，你为什么不换一件？" —— 看似吐槽，实则是心脏骤停后的掩饰"""


# ─── 常见误区（必须在改写中规避）──────────────────────────────
ANTI_PATTERNS = """\
以下是最常见的"伪文学"套路，改写时必须规避：

1. ❌ 万能忧伤环境：灰蒙蒙的天空、冷掉的食物、淅淅沥沥的雨、枯黄的落叶
   → 这些意象太通用，套在任何悲伤场景都成立，因此毫无信息量
   → 替代思路：找一个只属于此刻、只属于这个角色的具体细节

2. ❌ 生理反应堆砌：心跳加速、手心出汗、喉咙发紧、眼眶泛红
   → 这些是身体说明书，不是文学。八月长安几乎不写生理反应
   → 替代思路：用动作代替反应——"她把可乐罐捏扁了"比"她手心出汗"有力一百倍

3. ❌ 抒情式独白：内心独白大段铺陈，"我多么希望……"、"为什么总是……"
   → 八月长安的叙述者会自嘲，但不会无节制地抒情
   → 替代思路：把独白压缩成一个反问或一句轻描淡写

4. ❌ 比喻过载：连续使用多个比喻修饰同一个情绪
   → 一个精准的比喻胜过十个华丽的比喻
   → 替代思路：找到那个"对的"比喻就停下来"""


def _build_examples_block(results: list[dict]) -> str:
    """将检索结果格式化为 few-shot 示例块（含技法注释）"""
    if not results:
        return "（未检索到参考段落，请仅根据风格说明进行改写）"

    blocks = []
    for i, r in enumerate(results, 1):
        tags_str = "、".join(r["tags"]) if r["tags"] else "无"
        technique_note = r.get("technique_note", "")
        block = (
            f"【参考段落 {i}】\n"
            f"分类：{r['category']}\n"
            f"标签：{tags_str}\n"
            f"场景：{r['scene']}\n"
            f"技法注释（为什么这么写）：\n{technique_note}\n"
            f"原文：\n{r['text']}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_system_prompt() -> str:
    """构建系统提示词"""
    return (
        "你是一位精通八月长安文风的写作助手。"
        "你的任务是将用户输入的文本改写为八月长安的风格——"
        "不是套用文学模板，而是找到只属于此刻、只属于这个角色的具象细节来承载情感。"
        "你追求的是精准和意外，而不是华丽和通顺。"
    )


def _build_user_prompt(input_text: str, examples_block: str, max_words: int = None) -> str:
    """构建用户提示词"""
    if max_words and max_words > 0:
        length_requirement = f"7. 字数限制：改写后的文本请严格控制在{max_words}字以内，用最精炼的文字承载情绪"
    else:
        length_requirement = "7. 控制篇幅：改写后的长度应该是原文的2-5倍，不要过度展开"
    return f"""请将以下文本改写为八月长安的风格。

在动笔之前，请先在内心完成以下思考（不要输出思考过程）：
1. 这段文字的核心情绪是什么？（不是"难过"这种标签，而是具体的——"期待落空后的空白感"）
2. 这个情绪在什么具体的场景里最容易"漏出来"？（不是通用场景，而是只属于这个人的此刻）
3. 八月长安会怎么"藏"这个情绪？——找到一个意外的、有生活质感的具象画面
4. 仔细阅读参考段落的【技法注释】——理解八月长安为什么选择这个细节而非其他，AI容易在哪个环节丢失精髓。改写时不要模仿原文的表面特征（短句、日常意象、克制收尾），而要学习它的选择逻辑：为什么选这个细节、为什么在这个位置收手。

═══════════════════════════════════════
📌 八月长安风格核心特征
═══════════════════════════════════════

{STYLE_DESCRIPTION}

═══════════════════════════════════════
🚫 必须规避的套路
═══════════════════════════════════════

{ANTI_PATTERNS}

═══════════════════════════════════════
📖 参考段落（八月长安原文，请体会其手法）
═══════════════════════════════════════

{examples_block}

═══════════════════════════════════════
✏️ 待改写文本
═══════════════════════════════════════

{input_text}

═══════════════════════════════════════
🎯 改写要求
═══════════════════════════════════════

1. 保留原文的核心事件和情感走向
2. 找一个只属于此刻的具象细节来承载情绪——不要用通用意象
3. 如果原文很短（只有一句话），你需要创造一个有生活质感的微场景来展开
4. 情绪藏在缝隙里，不要直接说"失望/难过/开心"，让读者自己感受到
5. 叙述者的语气要有温度——可以自嘲、可以轻描淡写，但不要无节制抒情
6. 一个精准的比喻就够了，不要堆砌
{length_requirement}

请直接输出改写后的文本，不要添加任何解释或说明："""


def rewrite_to_august_style(
    input_text: str,
    api_key: str = None,
    model: str = DEFAULT_MODEL,
    category_hint: str = None,
    tags_hint: str = None,
    top_k: int = 3,
    base_url: str = None,
    max_words: int = None,
) -> str:
    """
    将输入文本改写为八月长安风格（支持多模型提供商）

    Parameters
    ----------
    input_text : str
        需要改写的原始文本
    api_key : str
        API Key（智谱/DeepSeek/阿里/Kimi 均可）
    model : str
        使用的模型，默认 glm-4-flash。
        支持：glm-4-flash, glm-4, deepseek-chat, qwen-plus, qwen-max, moonshot-v1-8k
    category_hint : str, optional
        分类提示，用于定向检索参考段落
    tags_hint : str, optional
        标签提示，用于定向检索参考段落
    top_k : int
        检索的参考段落数量
    base_url : str, optional
        API 基础地址。不传则根据 model 名称自动匹配 MODEL_CONFIG。

    Returns
    -------
    str
        改写后的文本
    """
    key = api_key or os.environ.get("ZHIPU_API_KEY", "")
    if not key:
        raise ValueError("缺少 API Key，请设置环境变量或传入 api_key 参数")

    # 自动匹配 base_url（支持多模型提供商）
    if base_url is None:
        config = MODEL_CONFIG.get(model, {})
        base_url = config.get("base_url", ZHIPU_API_BASE)

    # 1. 检索风格参考段落
    results = search(
        query=input_text,
        category_filter=category_hint,
        tags_filter=tags_hint,
        top_k=top_k,
    )

    # 2. 格式化示例
    examples_block = _build_examples_block(results)

    # 3. 构建提示词
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(input_text, examples_block, max_words=max_words)

    # 4. 调用模型 API
    client = OpenAI(api_key=key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=2000,
    )

    return response.choices[0].message.content.strip()


def get_prompt_only(
    input_text: str,
    category_hint: str = None,
    tags_hint: str = None,
    top_k: int = 3,
    max_words: int = None,
) -> str:
    """
    仅生成 prompt（不调用 API），用于展示检索过程

    Returns
    -------
    str
        构建好的完整 prompt
    """
    results = search(
        query=input_text,
        category_filter=category_hint,
        tags_filter=tags_hint,
        top_k=top_k,
    )
    examples_block = _build_examples_block(results)
    return _build_user_prompt(input_text, examples_block, max_words=max_words)


# ─── CoT 改写（方案2：思考链） ─────────────────────────────────


def _build_cot_analysis_prompt(input_text: str, examples_block: str) -> str:
    """构建CoT分析提示词（第一步：思考，不生成改写）"""
    text_len = len(input_text)
    target_min = int(text_len * 1.2)
    target_max = int(text_len * 2.0)

    return f"""请对以下文本进行八月长安风格的改写分析。只输出分析，不要生成改写结果。

═══════════════════════════════════════
第一步：原文解构
═══════════════════════════════════════
- 核心情绪（不是"难过"这种标签，而是具体的情绪状态，如"期待落空后的空白感"）
- 场景类型与关键信息节点（列出原文中所有信息点，标注哪些必须保留）
- 原文已有的好细节（不需要改动的部分）
- 原文的节奏特征（长短句、停顿、留白）

═══════════════════════════════════════
第二步：技法匹配
═══════════════════════════════════════
从参考段落和技法注释中，提取2-3个最适用于本次改写的技法：
- 技法名称
- 为什么适用于这个场景
- 具体怎么用（指明用在原文的哪个位置，用什么物件/感官）

═══════════════════════════════════════
第三步：改写策略
═══════════════════════════════════════
- 保留什么：列出必须原样保留的句子（特别注意：角色直接内心独白、对角色的直接吐槽——这些不能被替换为叙述者评论）
- 添加什么：列出要新增的具象细节（最多2个，每个说明用什么物件、什么感官、承载什么情绪）
- 在哪里收手：明确指出改写的最后一个情绪落点，在此之后不要继续展开
- 节奏安排：哪里用短句、哪里可以稍长、哪里必须停顿
- 字数目标：原文约{text_len}字，改写目标{target_min}-{target_max}字，不要超过{target_max}字

═══════════════════════════════════════
风格核心特征
═══════════════════════════════════════

{STYLE_DESCRIPTION}

═══════════════════════════════════════
必须规避的套路
═══════════════════════════════════════

{ANTI_PATTERNS}

═══════════════════════════════════════
参考段落（含技法注释）
═══════════════════════════════════════

{examples_block}

═══════════════════════════════════════
待改写文本
═══════════════════════════════════════

{input_text}

请按以上三步输出分析，不要生成改写结果。"""


def _build_cot_generation_prompt(input_text: str, analysis: str, max_words: int = None) -> str:
    """构建CoT生成提示词（第二步：基于分析策略生成）"""
    length_note = ""
    if max_words and max_words > 0:
        length_note = f"\n字数限制：{max_words}字以内。"

    return f"""基于以下分析策略，将文本改写为八月长安风格。

【改写分析策略】
{analysis}

【原文】
{input_text}

【执行要求】
1. 严格按照分析中的"改写策略"执行——只添加策略中列出的细节，不要自行增加环境描写或比喻
2. 分析中标注"必须保留"的句子，原样保留，不要改写或替换
3. 在分析中指定的"最后一个情绪落点"处收手，不要继续展开
4. 字数控制在分析中预估的范围内{length_note}
5. 情绪藏在缝隙里，不要直接说出来
6. 一个精准的比喻就够了，不要堆砌

请直接输出改写后的文本，不要添加任何解释："""


def rewrite_with_cot(
    input_text: str,
    api_key: str = None,
    model: str = DEFAULT_MODEL,
    category_hint: str = None,
    tags_hint: str = None,
    top_k: int = 3,
    base_url: str = None,
    max_words: int = None,
) -> dict:
    """
    CoT改写：两步思考链

    第一步：分析原文 → 提取技法 → 制定策略（不生成文本）
    第二步：基于策略生成改写文本

    与 rewrite_to_august_style 的区别：
    - 原方法：一次性生成，模型在生成时可能"忘记"技法注释的指导
    - CoT方法：先分析（低温0.3，严谨），再生成（高温0.8，创意），分析结论作为生成约束
    - 特别解决：篇幅失控、添加未计划的环境描写、替换角色直接独白为叙述者评论

    Returns
    -------
    dict
        {"thinking": "分析过程（可展示给用户）", "result": "改写结果"}
    """
    key = api_key or os.environ.get("ZHIPU_API_KEY", "")
    if not key:
        raise ValueError("缺少 API Key，请设置环境变量或传入 api_key 参数")

    if base_url is None:
        config = MODEL_CONFIG.get(model, {})
        base_url = config.get("base_url", ZHIPU_API_BASE)

    # 检索参考段落
    results = search(
        query=input_text,
        category_filter=category_hint,
        tags_filter=tags_hint,
        top_k=top_k,
    )
    examples_block = _build_examples_block(results)

    client = OpenAI(api_key=key, base_url=base_url)

    # Step 1: 分析（低温保证严谨）
    analysis_prompt = _build_cot_analysis_prompt(input_text, examples_block)
    analysis_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一位文学写作分析师，精通八月长安风格。你的任务是分析文本并制定改写策略，不生成改写结果。分析要具体、可执行，不要泛泛而谈。"},
            {"role": "user", "content": analysis_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    thinking = analysis_response.choices[0].message.content.strip()

    # Step 2: 基于策略生成（高温保证创意）
    generation_prompt = _build_cot_generation_prompt(input_text, thinking, max_words=max_words)
    gen_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": generation_prompt},
        ],
        temperature=0.8,
        max_tokens=2000,
    )
    result = gen_response.choices[0].message.content.strip()

    return {"thinking": thinking, "result": result}


if __name__ == "__main__":
    test_text = "她很难过，因为他没有回复消息。她一直在等，但手机始终没有响。她觉得自己很傻。"
    result = rewrite_to_august_style(test_text)
    print(result)
