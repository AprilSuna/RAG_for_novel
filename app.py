"""
AI文学写作能力评测与优化系统 - Streamlit 主界面
================================================
核心功能：
  - 风格检索：语义搜索八月长安风格段落（支持分类/标签/情绪过滤）
  - 风格改写：多模型对比改写，输入文字调用GLM改写为八月长安风格
  - 能力评测：对改写结果进行9维度评分，将"写得好不好"量化为可度量指标

技术栈：智谱 API (embedding-3 + GLM-4) + numpy 余弦相似度
零编译依赖，macOS 原生可运行。
"""

import json
import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embedding import EXCERPTS_PATH, build_index, index_exists, load_index
from src.generation import rewrite_to_august_style, get_prompt_only, rewrite_with_cot
from src.retrieval import list_all_tags, list_categories, list_emotions, search
from src.evaluation import evaluate_text, format_evaluation_result

# ─── 创作工坊数据路径 ────────────────────────────────────────
NOVEL_DIR = os.path.join(PROJECT_ROOT, "data", "novel")
CHARACTERS_PATH = os.path.join(NOVEL_DIR, "characters.json")
RELATIONSHIPS_PATH = os.path.join(NOVEL_DIR, "relationships.json")
STYLE_NOTES_PATH = os.path.join(NOVEL_DIR, "style_notes.json")
MANUSCRIPTS_DIR = os.path.join(NOVEL_DIR, "manuscripts")
MANUSCRIPTS_META_PATH = os.path.join(NOVEL_DIR, "manuscripts.json")


def load_novel_data(path):
    """加载创作工坊数据"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_novel_data(path, data):
    """保存创作工坊数据"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_manuscript(filename):
    """读取稿件正文"""
    filepath = os.path.join(MANUSCRIPTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_manuscript(filename, content):
    """保存稿件正文"""
    os.makedirs(MANUSCRIPTS_DIR, exist_ok=True)
    filepath = os.path.join(MANUSCRIPTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


# ─── API Key 本地持久化 ──────────────────────────────────────
SAVED_KEYS_PATH = os.path.join(PROJECT_ROOT, "api_keys.json")


def load_saved_keys():
    """从 Streamlit secrets 或本地文件读取已保存的 API Key"""
    keys = {}
    # 先读本地文件
    if os.path.exists(SAVED_KEYS_PATH):
        try:
            with open(SAVED_KEYS_PATH, "r", encoding="utf-8") as f:
                keys = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # Streamlit Cloud secrets 优先（云端部署用）
    try:
        for k in ["zhipu", "deepseek", "kimi", "minimax"]:
            if k in st.secrets:
                keys[k] = st.secrets[k]
    except Exception:
        pass
    return keys


def save_keys_to_local(keys_dict):
    """将 API Key 保存到本地文件"""
    with open(SAVED_KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(keys_dict, f, ensure_ascii=False, indent=2)


# ─── 页面配置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="文风工坊 - 创作工作台",
    page_icon="📚",
    layout="wide",
)


@st.cache_data
def load_excerpt_count() -> int:
    with open(EXCERPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


# ── 禁用 Streamlit 内置 "C" 快捷键（清除缓存），避免与 Command+C 冲突 ──
import streamlit.components.v1 as components
components.html(
    """
<script>
(function() {
    var doc = window.parent.document || window.document;
    doc.addEventListener('keydown', function(e) {
        if ((e.key === 'c' || e.key === 'C') && (e.metaKey || e.ctrlKey)) {
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, true);
})();
</script>
""",
    height=0,
)


# ─── 侧边栏 ─────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 文风工坊")
    st.markdown("##### 八月长安文风拆解 · 创作工作台")
    st.markdown("---")

    # API Key 输入（从本地文件自动填充）
    st.markdown("### 🔑 API Key 配置")
    _saved = load_saved_keys()

    api_key = st.text_input(
        "智谱 API Key（必需）",
        type="password",
        value=_saved.get("zhipu", ""),
        placeholder="open.bigmodel.cn 注册获取",
        help="用于风格检索和评测，必填",
    )
    os.environ["ZHIPU_API_KEY"] = api_key

    deepseek_key = st.text_input(
        "DeepSeek API Key（可选）",
        type="password",
        value=_saved.get("deepseek", ""),
        placeholder="platform.deepseek.com 注册获取",
    )
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key

    kimi_key = st.text_input(
        "Kimi API Key（可选）",
        type="password",
        value=_saved.get("kimi", ""),
        placeholder="platform.moonshot.cn 注册获取",
    )
    os.environ["MOONSHOT_API_KEY"] = kimi_key

    minimax_key = st.text_input(
        "MiniMax API Key（可选）",
        type="password",
        value=_saved.get("minimax", ""),
        placeholder="platform.minimaxi.com 注册获取",
    )
    os.environ["MINIMAX_API_KEY"] = minimax_key

    # 💾 记住API Key：勾选后保存到本地，下次启动自动填充
    remember_keys = st.checkbox("💾 记住 API Key（保存到本地，下次自动填充）", value=True)
    if remember_keys:
        save_keys_to_local({
            "zhipu": api_key,
            "deepseek": deepseek_key,
            "kimi": kimi_key,
            "minimax": minimax_key,
        })

    st.markdown("---")

    # 构建可用模型列表（基于已配置的 API Key）
    available_models = []
    if api_key:
        available_models.append(("glm-4-flash", "GLM-4-Flash（免费）"))
        available_models.append(("glm-4", "GLM-4（智谱）"))
    if deepseek_key:
        available_models.append(("deepseek-chat", "DeepSeek-V3"))
    if kimi_key:
        available_models.append(("moonshot-v1-8k", "Kimi"))
    if minimax_key:
        available_models.append(("MiniMax-M1", "MiniMax-M1"))

    # 模型 → API Key 映射
    model_api_keys = {}
    if api_key:
        model_api_keys["glm-4-flash"] = api_key
        model_api_keys["glm-4"] = api_key
    if deepseek_key:
        model_api_keys["deepseek-chat"] = deepseek_key
    if kimi_key:
        model_api_keys["moonshot-v1-8k"] = kimi_key
    if minimax_key:
        model_api_keys["MiniMax-M1"] = minimax_key

    # 评测模型（固定使用 GLM-4，确保评分稳定一致）
    eval_model_name = "glm-4"

    st.markdown("---")
    st.markdown(
        """
        🎯 **核心功能**
        - **风格检索**：语义搜索八月长安风格段落
        - **风格改写**：单模型/多模型对比改写
        - **能力评测**：9维度量化评分 + 改进建议
        - **创作工坊**：人物档案、关系图谱、风格笔记、小说存档

        📚 **数据来源**
        - 《最好的我们》精选摘录
        - 《不够勇敢的我们》创作档案

        ⚙️ **技术栈**
        - 智谱 embedding-3 + 多模型改写
        - GLM / DeepSeek / Kimi / MiniMax
        - numpy 余弦相似度（零编译依赖）
        """
    )

    excerpt_count = load_excerpt_count()
    st.markdown("---")
    st.metric(label="当前数据量", value=f"{excerpt_count} 段")


# ─── 检查 API Key ────────────────────────────────────────────
if not api_key:
    st.warning("⚠️ 请在左侧边栏输入智谱 API Key 才能使用本系统")
    st.markdown(
        """
        ### 获取 API Key 步骤：
        1. 访问 [智谱开放平台](https://open.bigmodel.cn/)
        2. 注册账号（新用户送 500 万 token）
        3. 进入「API Keys」页面创建 Key
        4. 复制 Key 粘贴到左侧输入框
        """
    )
    st.stop()


# ─── 初始化索引 ──────────────────────────────────────────────
@st.cache_resource
def init_index(key: str):
    if not index_exists():
        with st.spinner("正在调用智谱 API 构建向量索引（首次约需 10 秒）..."):
            return build_index(api_key=key, force_rebuild=True)
    else:
        return load_index()


@st.cache_data
def get_categories(key: str):
    return list_categories(api_key=key)


@st.cache_data
def get_tags(key: str):
    return list_all_tags(api_key=key)


@st.cache_data
def get_emotions(key: str):
    return list_emotions(api_key=key)


try:
    init_index(api_key)
    categories = get_categories(api_key)
    tags = get_tags(api_key)
    emotions = get_emotions(api_key)
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.info("请检查 API Key 是否正确，或点击下方按钮重建索引")
    if st.button("🔄 重建索引"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.stop()


with st.sidebar:
    with st.expander("查看所有分类"):
        for cat in categories:
            st.markdown(f"- {cat}")
    with st.expander("查看所有标签"):
        st.markdown("、".join(tags))
    with st.expander("查看所有情绪基调"):
        for emo in emotions:
            st.markdown(f"- {emo}")


# ─── 主界面 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 风格检索", "✍️ 风格改写", "📊 能力评测", "📖 创作工坊"])

# ========== Tab1: 风格检索 ==========
with tab1:
    st.header("🔍 风格检索")
    st.caption("输入描述，检索最相似的八月长安风格段落")

    col_query, col_filter = st.columns([2, 1])

    with col_query:
        query_text = st.text_area(
            "描述你想查找的风格特征或场景",
            placeholder="例如：暗恋中的克制与隐忍、用环境映射心理、未完成的对话...",
            height=100,
        )

    with col_filter:
        st.markdown("**过滤条件（可选）**")
        category_options = ["不限"] + categories
        selected_category = st.selectbox("按分类过滤", category_options)

        tag_options = ["不限"] + tags
        selected_tag = st.selectbox("按标签过滤", tag_options)

        emotion_options = ["不限"] + emotions
        selected_emotion = st.selectbox("按情绪过滤", emotion_options)

        top_k = st.slider("返回条数", min_value=1, max_value=7, value=3)

    if st.button("🔎 检索", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("请输入检索描述")
        else:
            cat_filter = None if selected_category == "不限" else selected_category
            tag_filter = None if selected_tag == "不限" else selected_tag
            emo_filter = None if selected_emotion == "不限" else selected_emotion

            with st.spinner("正在检索..."):
                results = search(
                    query=query_text,
                    category_filter=cat_filter,
                    tags_filter=tag_filter,
                    emotion_filter=emo_filter,
                    top_k=top_k,
                    api_key=api_key,
                )

            if not results:
                st.info("未找到匹配段落，请调整检索描述或过滤条件")
            else:
                st.success(f"找到 {len(results)} 条匹配段落")
                for i, r in enumerate(results, 1):
                    with st.container():
                        col_header1, col_header2 = st.columns([3, 1])
                        with col_header1:
                            st.markdown(f"### 段落 {i}")
                        with col_header2:
                            if r["distance"] is not None:
                                st.caption(f"语义距离: {r['distance']:.4f}")

                        tag_chips = " ".join(f"`{t}`" for t in r["tags"])
                        emotion_str = r.get("emotion", "")
                        st.markdown(
                            f"📂 **分类**: `{r['category']}` &nbsp;|&nbsp; "
                            f"🏷️ **标签**: {tag_chips}"
                        )
                        if emotion_str:
                            st.markdown(f"💬 **情绪**: `{emotion_str}`")
                        st.markdown(f"🎬 **场景**: {r['scene']}")

                        # 优先展示精华句
                        if r.get("highlight"):
                            st.markdown(f"> ✨ **精华句**：{r['highlight']}")

                        with st.expander("📖 查看完整原文", expanded=False):
                            st.markdown(r["text"])

                        st.markdown("---")

# ========== Tab2: 风格改写 ==========
with tab2:
    st.header("✍️ 风格改写")

    # 模型选择
    model_options = [m[0] for m in available_models]
    model_labels = dict(available_models)
    selected_models = []

    if not available_models:
        st.warning("⚠️ 请先在左侧边栏配置至少一个模型的 API Key")
    else:
        selected_models = st.multiselect(
            "选择改写模型（可多选进行对比）",
            options=model_options,
            format_func=lambda x: model_labels.get(x, x),
            default=[model_options[0]],
        )
        if len(selected_models) > 1:
            st.caption(f"🔗 多模型对比模式：将同时使用 {len(selected_models)} 个模型改写")
        elif len(selected_models) == 1:
            st.caption(f"当前模型：{model_labels.get(selected_models[0], selected_models[0])}")

    input_text = st.text_area(
        "输入需要改写的文字",
        placeholder="例如：她很难过，因为他没有回复消息。她一直在等，但手机始终没有响。",
        height=150,
    )

    # 字数统计（右对齐显示在文本框下方）
    _, col_count = st.columns([5, 1])
    with col_count:
        st.caption(f"📊 {len(input_text)} 字")

    col_words, _ = st.columns([1, 3])
    with col_words:
        max_words = st.number_input(
            "📝 字数上限",
            min_value=0,
            value=0,
            step=50,
            help="0表示不限制字数。设置后改写结果会控制在指定字数以内",
        )

    col_rewrite_filter1, col_rewrite_filter2, col_rewrite_filter3 = st.columns(3)
    with col_rewrite_filter1:
        rewrite_category = st.selectbox(
            "参考段落分类",
            ["不限"] + categories,
            key="rewrite_category",
        )
    with col_rewrite_filter2:
        rewrite_tag = st.selectbox(
            "参考段落标签",
            ["不限"] + tags,
            key="rewrite_tag",
        )
    with col_rewrite_filter3:
        rewrite_emotion = st.selectbox(
            "参考段落情绪",
            ["不限"] + emotions,
            key="rewrite_emotion",
        )

    col_cot, _ = st.columns([1, 3])
    with col_cot:
        use_cot = st.checkbox("🧠 CoT改写模式（思考链）", value=False, help="启用后改写分两步：先分析策略再生成，改写思路可展开查看。每次改写会调用两次API。")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        rewrite_btn = st.button("✨ 改写", type="primary", use_container_width=True)

    with col_btn2:
        show_prompt_btn = st.button("📋 查看Prompt", use_container_width=True)

    with col_btn3:
        auto_eval = st.checkbox("改写后自动评测", value=True)

    with col_btn4:
        clear_btn = st.button("🗑️ 清除缓存", use_container_width=True)

    # ── 初始化 session_state 缓存 ──
    if "rewrite_results_cache" not in st.session_state:
        st.session_state["rewrite_results_cache"] = {}
    if "prompt_cache" not in st.session_state:
        st.session_state["prompt_cache"] = None
    if "eval_results_cache" not in st.session_state:
        st.session_state["eval_results_cache"] = {}
    if "last_input_cache" not in st.session_state:
        st.session_state["last_input_cache"] = ""
    if "cot_thinking_cache" not in st.session_state:
        st.session_state["cot_thinking_cache"] = {}

    # ── 清除缓存 ──
    if clear_btn:
        st.session_state["rewrite_results_cache"] = {}
        st.session_state["prompt_cache"] = None
        st.session_state["eval_results_cache"] = {}
        st.session_state["last_input_cache"] = ""
        st.session_state["cot_thinking_cache"] = {}
        st.rerun()

    # ── 改写按钮：生成改写+prompt+评测，全部存入缓存 ──
    if rewrite_btn:
        if not input_text.strip():
            st.warning("请输入需要改写的文字")
        elif not selected_models:
            st.warning("请至少选择一个改写模型")
        else:
            cat_hint = None if rewrite_category == "不限" else rewrite_category
            tag_hint = None if rewrite_tag == "不限" else rewrite_tag
            emo_hint = None if rewrite_emotion == "不限" else rewrite_emotion

            # 生成并缓存 prompt（改写时自动生成，不用额外点）
            prompt = get_prompt_only(
                input_text=input_text,
                category_hint=cat_hint,
                tags_hint=tag_hint,
                top_k=3,
                max_words=max_words if max_words > 0 else None,
            )
            st.session_state["prompt_cache"] = prompt
            st.session_state["last_input_cache"] = input_text

            # 改写
            rewrite_results = {}
            cot_thinking = {}
            for model_name in selected_models:
                model_key = model_api_keys.get(model_name, api_key)
                model_label = model_labels.get(model_name, model_name)
                if use_cot:
                    with st.spinner(f"正在用 {model_label} 进行 CoT 分析并改写（两步调用）..."):
                        try:
                            cot_result = rewrite_with_cot(
                                input_text=input_text,
                                api_key=model_key,
                                model=model_name,
                                category_hint=cat_hint,
                                tags_hint=tag_hint,
                                top_k=3,
                                max_words=max_words if max_words > 0 else None,
                            )
                            rewrite_results[model_name] = cot_result.get("result", "")
                            cot_thinking[model_name] = cot_result.get("thinking", "")
                        except Exception as e:
                            st.error(f"{model_label} CoT改写失败：{e}")
                            rewrite_results[model_name] = None
                            cot_thinking[model_name] = None
                else:
                    with st.spinner(f"正在用 {model_label} 检索参考段落并改写..."):
                        try:
                            result = rewrite_to_august_style(
                                input_text=input_text,
                                api_key=model_key,
                                model=model_name,
                                category_hint=cat_hint,
                                tags_hint=tag_hint,
                                top_k=3,
                                max_words=max_words if max_words > 0 else None,
                            )
                            rewrite_results[model_name] = result
                        except Exception as e:
                            st.error(f"{model_label} 改写失败：{e}")
                            rewrite_results[model_name] = None
            st.session_state["rewrite_results_cache"] = rewrite_results
            st.session_state["cot_thinking_cache"] = cot_thinking

            # 自动评测
            eval_results = {}
            if auto_eval:
                for model_name, result in rewrite_results.items():
                    if result:
                        with st.spinner(f"正在评测 {model_labels.get(model_name, model_name)} 的改写结果..."):
                            try:
                                eval_result = evaluate_text(
                                    original_text=input_text,
                                    rewritten_text=result,
                                    api_key=api_key,
                                    model=eval_model_name,
                                )
                                eval_results[model_name] = eval_result
                            except Exception as e:
                                eval_results[model_name] = None
            st.session_state["eval_results_cache"] = eval_results

    # ── 查看Prompt按钮：只更新prompt缓存，不影响改写结果 ──
    if show_prompt_btn:
        if not input_text.strip():
            st.warning("请输入需要改写的文字")
        else:
            cat_hint = None if rewrite_category == "不限" else rewrite_category
            tag_hint = None if rewrite_tag == "不限" else rewrite_tag
            prompt = get_prompt_only(
                input_text=input_text,
                category_hint=cat_hint,
                tags_hint=tag_hint,
                top_k=3,
                max_words=max_words if max_words > 0 else None,
            )
            st.session_state["prompt_cache"] = prompt

    # ── 展示缓存的改写结果（始终显示，不依赖按钮）──
    cached_rewrite = st.session_state.get("rewrite_results_cache", {})
    cached_eval = st.session_state.get("eval_results_cache", {})

    if cached_rewrite:
        cached_thinking = st.session_state.get("cot_thinking_cache", {})
        if len(cached_rewrite) == 1:
            model_name = list(cached_rewrite.keys())[0]
            result = cached_rewrite[model_name]
            if result:
                st.success(f"改写完成（{model_labels.get(model_name, model_name)}）！")

                # CoT 改写思路展示
                if model_name in cached_thinking and cached_thinking[model_name]:
                    with st.expander("🧠 改写思路（CoT分析）", expanded=False):
                        st.markdown(cached_thinking[model_name])

                st.markdown("#### ✍️ 改写结果")
                st.markdown(result)

                # 同步到评测tab可用
                st.session_state["last_input"] = st.session_state.get("last_input_cache", input_text)
                st.session_state["last_rewrite"] = result
                st.session_state["last_model"] = model_name

                # 展示评测
                if model_name in cached_eval and cached_eval[model_name]:
                    st.markdown("---")
                    st.markdown(format_evaluation_result(cached_eval[model_name]))
        else:
            st.success("多模型改写完成！")
            cached_thinking = st.session_state.get("cot_thinking_cache", {})
            cols = st.columns(len(cached_rewrite))
            for col, (model_name, result) in zip(cols, cached_rewrite.items()):
                with col:
                    st.markdown(f"#### 🤖 {model_labels.get(model_name, model_name)}")
                    if result:
                        if model_name in cached_thinking and cached_thinking[model_name]:
                            with st.expander("🧠 改写思路", expanded=False):
                                st.markdown(cached_thinking[model_name])
                        st.markdown(result)
                    else:
                        st.error("改写失败")

            # 同步第一个成功结果到评测tab
            for model_name, result in cached_rewrite.items():
                if result:
                    st.session_state["last_input"] = st.session_state.get("last_input_cache", input_text)
                    st.session_state["last_rewrite"] = result
                    st.session_state["last_model"] = model_name
                    break

            # 展示对比评测
            if cached_eval:
                st.markdown("---")
                st.markdown("### 📊 对比评测")
                for model_name, result in cached_rewrite.items():
                    if not result or model_name not in cached_eval or not cached_eval[model_name]:
                        continue
                    st.markdown(f"#### {model_labels.get(model_name, model_name)} 评测")
                    st.markdown(format_evaluation_result(cached_eval[model_name]))
                    st.markdown("---")

    # ── 展示缓存的Prompt（始终显示，不依赖按钮）──
    cached_prompt = st.session_state.get("prompt_cache")
    if cached_prompt:
        with st.expander("📋 查看检索 Prompt（参考段落+风格说明）", expanded=False):
            st.code(cached_prompt, language="markdown")

# ========== Tab3: 能力评测 ==========
with tab3:
    st.header("📊 写作能力评测")
    st.caption('青春文学评测框架 v2 — 9维度5档评分，通用维度必评 + 场景维度可N/A')

    st.markdown("#### 评测维度")
    st.markdown("""
    **通用维度（必评）**

    | 维度 | 评测内容 |
    |------|---------|
    | 具象精准度 | 细节是否「准」而非「美」，反浪漫化 |
    | 情感间接性 | 情绪是否藏在缝隙里，通过错位场景间接传达 |
    | 克制留白程度 | 是否做到了「不写满」，留白让读者补全 |
    | 节奏控制 | 叙事节奏是否有效，长短句配合、时间压缩 |
    | 青春期心理真实性 | 角色反应是否符合那个年纪，情绪强度与表达能力的gap |

    **场景维度（可标记N/A）**

    | 维度 | 评测内容 | N/A条件 |
    |------|---------|---------|
    | 对话潜台词密度 | 对话是否有言外之意 | 无对话 |
    | 环境映射质量 | 环境与心理是否错位 | 无环境描写 |
    | 叙述者声音 | 是否有温度的旁观者视角 | 纯对话无叙述层 |
    | 幽默反差运用 | 是否用幽默包裹沉重 | 不涉及幽默 |
    """)

    st.markdown("---")

    # 如果有上一次改写的结果，预填
    eval_original = st.text_area(
        "原始文本",
        value=st.session_state.get("last_input", ""),
        placeholder="输入原始文本（改写前）",
        height=100,
        key="eval_original",
    )

    eval_rewritten = st.text_area(
        "待评测文本",
        value=st.session_state.get("last_rewrite", ""),
        placeholder="输入待评测的文本（改写后或任意文本）",
        height=150,
        key="eval_rewritten",
    )

    if st.button("📊 开始评测", type="primary", use_container_width=True):
        if not eval_original.strip() or not eval_rewritten.strip():
            st.warning("请输入原始文本和待评测文本")
        else:
            with st.spinner("正在检索参考标准并进行评测..."):
                try:
                    eval_result = evaluate_text(
                        original_text=eval_original,
                        rewritten_text=eval_rewritten,
                        api_key=api_key,
                        model=eval_model_name,
                    )
                    st.markdown(format_evaluation_result(eval_result))

                    # 提供导出功能
                    st.markdown("---")
                    st.markdown("#### 📋 评测结果（JSON）")
                    st.json(eval_result)

                except Exception as e:
                    st.error(f"评测失败：{e}")

# ========== Tab4: 创作工坊 ==========
with tab4:
    st.header("📖 创作工坊")
    st.caption("《不够勇敢的我们》创作工作台 — 人物档案 · 关系图谱 · 风格笔记 · 小说存档")

    workshop_mode = st.radio(
        "选择模块",
        ["👤 人物档案", "🔗 关系图谱", "📝 风格笔记", "📚 小说存档"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ---------- 人物档案 ----------
    if "人物档案" in workshop_mode:
        char_data = load_novel_data(CHARACTERS_PATH)

        with st.expander("➕ 添加新角色"):
            with st.form("add_character_form"):
                col1, col2 = st.columns(2)
                with col1:
                    c_name = st.text_input("角色名")
                    c_role = st.text_input("角色定位（如：女主角、男一、闺蜜）")
                    c_age = st.text_input("年龄/年级")
                with col2:
                    c_tags = st.text_input("性格标签（逗号分隔）")
                    c_emotions = st.text_input("关联情绪基调（逗号分隔）")
                    c_linked_tags = st.text_input("关联写作标签（逗号分隔）")

                c_profile = st.text_area("性格底色描述")
                c_backstory = st.text_area("人物小传/背景")
                c_arc = st.text_input("成长弧线")
                c_conflict = st.text_area("内心冲突")

                if st.form_submit_button("添加角色"):
                    if c_name.strip():
                        new_char = {
                            "id": max([c["id"] for c in char_data], default=0) + 1,
                            "name": c_name.strip(),
                            "role": c_role.strip(),
                            "age": c_age.strip(),
                            "tags": [t.strip() for t in c_tags.split(",") if t.strip()],
                            "profile": c_profile.strip(),
                            "backstory": c_backstory.strip(),
                            "arc": c_arc.strip(),
                            "inner_conflict": c_conflict.strip(),
                            "linked_emotions": [t.strip() for t in c_emotions.split(",") if t.strip()],
                            "linked_tags": [t.strip() for t in c_linked_tags.split(",") if t.strip()],
                        }
                        char_data.append(new_char)
                        save_novel_data(CHARACTERS_PATH, char_data)
                        st.success("角色已添加！")
                        st.rerun()
                    else:
                        st.warning("请输入角色名")

        for char in char_data:
            editing = st.session_state.get("editing_char_id") == char["id"]

            if editing:
                # ── 编辑模式 ──
                with st.container():
                    st.markdown(f"### ✏️ 编辑角色：{char['name']}")
                    with st.form(f"edit_char_form_{char['id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            e_name = st.text_input("角色名", value=char.get("name", ""))
                            e_role = st.text_input("角色定位", value=char.get("role", ""))
                            e_age = st.text_input("年龄/年级", value=char.get("age", ""))
                        with col2:
                            e_tags = st.text_input("性格标签（逗号分隔）", value=",".join(char.get("tags", [])))
                            e_emotions = st.text_input("关联情绪基调（逗号分隔）", value=",".join(char.get("linked_emotions", [])))
                            e_linked_tags = st.text_input("关联写作标签（逗号分隔）", value=",".join(char.get("linked_tags", [])))

                        e_profile = st.text_area("性格底色描述", value=char.get("profile", ""))
                        e_backstory = st.text_area("人物小传/背景", value=char.get("backstory", ""))
                        e_arc = st.text_input("成长弧线", value=char.get("arc", ""))
                        e_conflict = st.text_area("内心冲突", value=char.get("inner_conflict", ""))

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_btn = st.form_submit_button("💾 保存修改", use_container_width=True, type="primary")
                        with col_cancel:
                            cancel_btn = st.form_submit_button("取消", use_container_width=True)

                    if save_btn:
                        char["name"] = e_name.strip()
                        char["role"] = e_role.strip()
                        char["age"] = e_age.strip()
                        char["tags"] = [t.strip() for t in e_tags.split(",") if t.strip()]
                        char["profile"] = e_profile.strip()
                        char["backstory"] = e_backstory.strip()
                        char["arc"] = e_arc.strip()
                        char["inner_conflict"] = e_conflict.strip()
                        char["linked_emotions"] = [t.strip() for t in e_emotions.split(",") if t.strip()]
                        char["linked_tags"] = [t.strip() for t in e_linked_tags.split(",") if t.strip()]
                        save_novel_data(CHARACTERS_PATH, char_data)
                        st.session_state["editing_char_id"] = None
                        st.success("修改已保存！")
                        st.rerun()

                    if cancel_btn:
                        st.session_state["editing_char_id"] = None
                        st.rerun()

                    st.markdown("---")

            else:
                # ── 展示模式 ──
                with st.container():
                    col_header, col_edit, col_del = st.columns([4, 1, 1])
                    with col_header:
                        tag_chips = " ".join(f"`{t}`" for t in char.get("tags", []))
                        st.markdown(f"### {char['name']}  `{char.get('role', '')}`")
                        if tag_chips:
                            st.markdown(f"**标签**: {tag_chips}")
                    with col_edit:
                        if st.button("✏️ 编辑", key=f"edit_char_{char['id']}"):
                            st.session_state["editing_char_id"] = char["id"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_char_{char['id']}"):
                            char_data = [c for c in char_data if c["id"] != char["id"]]
                            save_novel_data(CHARACTERS_PATH, char_data)
                            st.rerun()

                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        if char.get("age"):
                            st.markdown(f"**年龄**: {char['age']}")
                        if char.get("arc"):
                            st.markdown(f"**成长弧线**: {char['arc']}")
                    with col_info2:
                        if char.get("linked_emotions"):
                            st.markdown("**关联情绪**: " + "、".join(char["linked_emotions"]))
                        if char.get("linked_tags"):
                            st.markdown("**关联标签**: " + "、".join(char["linked_tags"]))

                    if char.get("profile"):
                        st.markdown(f"**性格底色**: {char['profile']}")
                    if char.get("backstory"):
                        with st.expander("人物小传"):
                            st.markdown(char["backstory"])
                    if char.get("inner_conflict"):
                        st.markdown(f"**内心冲突**: {char['inner_conflict']}")

                    # 检索参考段落
                    linked_tags = char.get("linked_tags", [])
                    linked_emotions = char.get("linked_emotions", [])
                    if linked_tags or linked_emotions:
                        if st.button(f"检索「{char['name']}」相关参考段落", key=f"search_char_{char['id']}"):
                            query_parts = []
                            if linked_tags:
                                query_parts.append("、".join(linked_tags))
                            if linked_emotions:
                                query_parts.append("、".join(linked_emotions))
                            query = " ".join(query_parts)

                            with st.spinner("正在检索参考段落..."):
                                results = search(
                                    query=query,
                                    emotion_filter=linked_emotions[0] if linked_emotions else None,
                                    top_k=5,
                                    api_key=api_key,
                                )

                            if results:
                                st.success(f"找到 {len(results)} 条参考段落")
                                for i, r in enumerate(results, 1):
                                    tag_str = " ".join(f"`{t}`" for t in r["tags"])
                                    st.markdown(f"**段落 {i}** `{r['category']}` | {tag_str} | `{r.get('emotion', '')}`")
                                    if r.get("highlight"):
                                        st.markdown(f"> {r['highlight']}")
                                    with st.expander("查看原文"):
                                        st.markdown(r["text"])
                            else:
                                st.info("未找到匹配段落")

                    st.markdown("---")

    # ---------- 关系图谱 ----------
    elif "关系图谱" in workshop_mode:
        rel_data = load_novel_data(RELATIONSHIPS_PATH)

        with st.expander("➕ 添加新关系"):
            with st.form("add_rel_form"):
                col1, col2 = st.columns(2)
                with col1:
                    r_a = st.text_input("角色A")
                    r_type = st.text_input("关系类型（如：暗恋、闺蜜、情敌）")
                with col2:
                    r_b = st.text_input("角色B")
                    r_key_moments = st.text_input("关键时刻（逗号分隔）")

                r_dynamics = st.text_area("关系动力学描述")
                r_tension = st.text_area("关系张力")

                if st.form_submit_button("添加关系"):
                    if r_a.strip() and r_b.strip():
                        new_rel = {
                            "id": max([r["id"] for r in rel_data], default=0) + 1,
                            "character_a": r_a.strip(),
                            "character_b": r_b.strip(),
                            "type": r_type.strip(),
                            "dynamics": r_dynamics.strip(),
                            "tension": r_tension.strip(),
                            "key_moments": [m.strip() for m in r_key_moments.split(",") if m.strip()],
                        }
                        rel_data.append(new_rel)
                        save_novel_data(RELATIONSHIPS_PATH, rel_data)
                        st.success("关系已添加！")
                        st.rerun()
                    else:
                        st.warning("请输入两个角色名")

        for rel in rel_data:
            editing = st.session_state.get("editing_rel_id") == rel["id"]

            if editing:
                with st.container():
                    st.markdown(f"### ✏️ 编辑关系：{rel['character_a']} <-> {rel['character_b']}")
                    with st.form(f"edit_rel_form_{rel['id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            e_a = st.text_input("角色A", value=rel.get("character_a", ""))
                            e_type = st.text_input("关系类型", value=rel.get("type", ""))
                        with col2:
                            e_b = st.text_input("角色B", value=rel.get("character_b", ""))
                            e_key_moments = st.text_input("关键时刻（逗号分隔）", value=",".join(rel.get("key_moments", [])))

                        e_dynamics = st.text_area("关系动力学描述", value=rel.get("dynamics", ""))
                        e_tension = st.text_area("关系张力", value=rel.get("tension", ""))

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_btn = st.form_submit_button("💾 保存修改", use_container_width=True, type="primary")
                        with col_cancel:
                            cancel_btn = st.form_submit_button("取消", use_container_width=True)

                    if save_btn:
                        rel["character_a"] = e_a.strip()
                        rel["character_b"] = e_b.strip()
                        rel["type"] = e_type.strip()
                        rel["dynamics"] = e_dynamics.strip()
                        rel["tension"] = e_tension.strip()
                        rel["key_moments"] = [m.strip() for m in e_key_moments.split(",") if m.strip()]
                        save_novel_data(RELATIONSHIPS_PATH, rel_data)
                        st.session_state["editing_rel_id"] = None
                        st.success("修改已保存！")
                        st.rerun()

                    if cancel_btn:
                        st.session_state["editing_rel_id"] = None
                        st.rerun()

                    st.markdown("---")

            else:
                with st.container():
                    col_header, col_edit, col_del = st.columns([4, 1, 1])
                    with col_header:
                        st.markdown(f"### {rel['character_a']} <-> {rel['character_b']}  `{rel.get('type', '')}`")
                    with col_edit:
                        if st.button("✏️ 编辑", key=f"edit_rel_{rel['id']}"):
                            st.session_state["editing_rel_id"] = rel["id"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_rel_{rel['id']}"):
                            rel_data = [r for r in rel_data if r["id"] != rel["id"]]
                            save_novel_data(RELATIONSHIPS_PATH, rel_data)
                            st.rerun()

                    if rel.get("dynamics"):
                        st.markdown(f"**动力学**: {rel['dynamics']}")
                    if rel.get("tension"):
                        st.markdown(f"**张力**: {rel['tension']}")
                    if rel.get("key_moments"):
                        st.markdown("**关键时刻**:")
                        for m in rel["key_moments"]:
                            st.markdown(f"- {m}")

                    st.markdown("---")

    # ---------- 风格笔记 ----------
    elif "风格笔记" in workshop_mode:
        note_data = load_novel_data(STYLE_NOTES_PATH)

        with st.expander("➕ 添加新笔记"):
            with st.form("add_note_form"):
                n_topic = st.text_input("主题（如：如何写XXX）")
                n_insight = st.text_area("洞察/心得")
                n_refs = st.text_input("参考摘录ID（逗号分隔，如：3, 5, 16）")
                n_applied = st.text_input("应用在哪一章/场景")
                n_status = st.selectbox("状态", ["构思中", "已实践", "待验证"])

                if st.form_submit_button("添加笔记"):
                    if n_topic.strip():
                        new_note = {
                            "id": max([n["id"] for n in note_data], default=0) + 1,
                            "topic": n_topic.strip(),
                            "insight": n_insight.strip(),
                            "reference_excerpts": [int(x.strip()) for x in n_refs.split(",") if x.strip().isdigit()],
                            "applied_in": n_applied.strip(),
                            "status": n_status,
                        }
                        note_data.append(new_note)
                        save_novel_data(STYLE_NOTES_PATH, note_data)
                        st.success("笔记已添加！")
                        st.rerun()
                    else:
                        st.warning("请输入主题")

        for note in note_data:
            editing = st.session_state.get("editing_note_id") == note["id"]

            if editing:
                with st.container():
                    st.markdown(f"### ✏️ 编辑笔记：{note['topic']}")
                    with st.form(f"edit_note_form_{note['id']}"):
                        e_topic = st.text_input("主题", value=note.get("topic", ""))
                        e_insight = st.text_area("洞察/心得", value=note.get("insight", ""))
                        e_refs = st.text_input("参考摘录ID（逗号分隔）", value=",".join(str(r) for r in note.get("reference_excerpts", [])))
                        e_applied = st.text_input("应用在哪一章/场景", value=note.get("applied_in", ""))
                        e_status = st.selectbox("状态", ["构思中", "已实践", "待验证"], index=["构思中", "已实践", "待验证"].index(note.get("status", "构思中")))

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_btn = st.form_submit_button("💾 保存修改", use_container_width=True, type="primary")
                        with col_cancel:
                            cancel_btn = st.form_submit_button("取消", use_container_width=True)

                    if save_btn:
                        note["topic"] = e_topic.strip()
                        note["insight"] = e_insight.strip()
                        note["reference_excerpts"] = [int(x.strip()) for x in e_refs.split(",") if x.strip().isdigit()]
                        note["applied_in"] = e_applied.strip()
                        note["status"] = e_status
                        save_novel_data(STYLE_NOTES_PATH, note_data)
                        st.session_state["editing_note_id"] = None
                        st.success("修改已保存！")
                        st.rerun()

                    if cancel_btn:
                        st.session_state["editing_note_id"] = None
                        st.rerun()

                    st.markdown("---")

            else:
                with st.container():
                    col_header, col_edit, col_del = st.columns([4, 1, 1])
                    with col_header:
                        st.markdown(f"### {note['topic']}  `{note.get('status', '')}`")
                    with col_edit:
                        if st.button("✏️ 编辑", key=f"edit_note_{note['id']}"):
                            st.session_state["editing_note_id"] = note["id"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_note_{note['id']}"):
                            note_data = [n for n in note_data if n["id"] != note["id"]]
                            save_novel_data(STYLE_NOTES_PATH, note_data)
                            st.rerun()

                    if note.get("status"):
                        st.markdown(f"**状态**: `{note['status']}`")
                    if note.get("insight"):
                        st.markdown(f"**洞察**: {note['insight']}")
                    if note.get("reference_excerpts"):
                        ref_str = ", ".join(f"#{r}" for r in note["reference_excerpts"])
                        st.markdown(f"**参考摘录**: {ref_str}")
                    if note.get("applied_in"):
                        st.markdown(f"**应用位置**: {note['applied_in']}")

                    st.markdown("---")

    # ---------- 小说存档 ----------
    elif "小说存档" in workshop_mode:
        ms_meta = load_novel_data(MANUSCRIPTS_META_PATH)

        with st.expander("➕ 添加新稿件"):
            with st.form("add_manuscript_form"):
                col1, col2 = st.columns(2)
                with col1:
                    m_title = st.text_input("标题")
                    m_category = st.selectbox("类型", ["正文", "初稿", "大纲", "设定", "笔记"])
                with col2:
                    m_status = st.selectbox("状态", ["构思中", "写作中", "完成", "归档"])
                    m_notes = st.text_input("备注")

                m_content = st.text_area("正文内容（可直接粘贴或稍后编辑）", height=200)

                if st.form_submit_button("添加稿件"):
                    if m_title.strip():
                        import re
                        safe_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', m_title.strip()) + ".md"
                        new_ms = {
                            "id": max([m["id"] for m in ms_meta], default=0) + 1,
                            "title": m_title.strip(),
                            "filename": safe_name,
                            "category": m_category,
                            "status": m_status,
                            "word_count": len(m_content),
                            "notes": m_notes.strip(),
                        }
                        ms_meta.append(new_ms)
                        save_novel_data(MANUSCRIPTS_META_PATH, ms_meta)
                        save_manuscript(safe_name, m_content)
                        st.success("稿件已添加！")
                        st.rerun()
                    else:
                        st.warning("请输入标题")

        # 按类型分组展示
        categories_order = ["正文", "初稿", "大纲", "设定", "笔记"]
        for cat in categories_order:
            cat_items = [m for m in ms_meta if m.get("category") == cat]
            if not cat_items:
                continue

            st.markdown(f"#### {cat}（{len(cat_items)} 篇）")

            for ms in cat_items:
                with st.container():
                    col_header, col_edit, col_del = st.columns([4, 1, 1])
                    with col_header:
                        st.markdown(f"**{ms['title']}**  `{ms.get('status', '')}`")
                        if ms.get("word_count"):
                            st.caption(f"约 {ms['word_count']} 字 | {ms.get('notes', '')}")
                    with col_edit:
                        if st.button("✏️ 编辑", key=f"edit_ms_{ms['id']}"):
                            st.session_state["editing_ms_id"] = ms["id"]
                            st.rerun()
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_ms_{ms['id']}"):
                            # 删除文件
                            filepath = os.path.join(MANUSCRIPTS_DIR, ms["filename"])
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            # 删除元数据
                            ms_meta = [m for m in ms_meta if m["id"] != ms["id"]]
                            save_novel_data(MANUSCRIPTS_META_PATH, ms_meta)
                            st.rerun()

                    editing_ms = st.session_state.get("editing_ms_id") == ms["id"]

                    if editing_ms:
                        current_content = read_manuscript(ms["filename"])
                        with st.form(f"edit_ms_form_{ms['id']}"):
                            e_title = st.text_input("标题", value=ms.get("title", ""))
                            e_status = st.selectbox(
                                "状态",
                                ["构思中", "写作中", "完成", "归档"],
                                index=["构思中", "写作中", "完成", "归档"].index(ms.get("status", "构思中")),
                            )
                            e_notes = st.text_input("备注", value=ms.get("notes", ""))
                            e_content = st.text_area("正文", value=current_content, height=400)

                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                save_btn = st.form_submit_button("💾 保存修改", use_container_width=True, type="primary")
                            with col_cancel:
                                cancel_btn = st.form_submit_button("取消", use_container_width=True)

                        if save_btn:
                            ms["title"] = e_title.strip()
                            ms["status"] = e_status
                            ms["notes"] = e_notes.strip()
                            ms["word_count"] = len(e_content)
                            save_novel_data(MANUSCRIPTS_META_PATH, ms_meta)
                            save_manuscript(ms["filename"], e_content)
                            st.session_state["editing_ms_id"] = None
                            st.success("稿件已保存！")
                            st.rerun()

                        if cancel_btn:
                            st.session_state["editing_ms_id"] = None
                            st.rerun()
                    else:
                        content = read_manuscript(ms["filename"])
                        if content:
                            with st.expander("📖 查看正文", expanded=False):
                                st.markdown(content)

                    st.markdown("---")