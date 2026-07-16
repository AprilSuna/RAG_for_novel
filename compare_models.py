"""
多模型改写对比脚本
==================
使用相同的 prompt 和参考段落，调用不同主流大模型进行改写，
并用统一的评测模型（GLM-4）对结果打分，生成对比报告。

═══════════════════════════════════════════════════
快速开始
═══════════════════════════════════════════════════

1. 安装依赖（如已安装可跳过）：
   pip install openai numpy

2. 注册并获取 API Key（详见 docs/API注册指南.md）：
   - 智谱（必须，用于评测）：https://open.bigmodel.cn/
   - DeepSeek：https://platform.deepseek.com/
   - 阿里通义千问：https://bailian.console.aliyun.com/
   - 月之暗面 Kimi：https://platform.moonshot.cn/

3. 设置环境变量（在终端中执行）：
   export ZHIPU_API_KEY=你的智谱key
   export DEEPSEEK_API_KEY=你的deepseek_key
   export DASHSCOPE_API_KEY=你的阿里key
   export MOONSHOT_API_KEY=你的kimi_key

   （Windows 用 set 代替 export）

4. 运行：
   python compare_models.py

5. 查看结果：
   - 控制台输出对比摘要
   - output/ 目录下生成详细 markdown 报告 + JSON 原始数据

═══════════════════════════════════════════════════
"""

import os
import sys
import json
import time
from datetime import datetime

# 确保能导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generation import rewrite_to_august_style, get_prompt_only, MODEL_CONFIG
from src.evaluation import evaluate_text


# ─── 测试场景定义 ──────────────────────────────────────────────
# 每个场景包含：名称、分类提示、测试文本
# April 可以替换 TEST_SCENES 中的文本来自定义测试场景

TEST_SCENES = [
    {
        "name": "心理描写",
        "category": "心理与情感",
        "text": """夏知晴在许燃的QQ空间动态下面，把编辑了半小时的评论逐字删掉，点了一个赞。

那是凌晨零点十二分。

许燃的动态很长，第一段写高中时候和兄弟们打篮球、吃食堂、被罚抄笔记。写得散漫，像随手翻一本旧相册，想起什么就写什么。夏知晴一行一行往下拉，心跳越来越快——里面没有她。

直到最后一行。

"有一个女孩，她几乎陪伴了我一整个高中。现在，我终于放下，该向前走了。"

她盯着那句话看了很久。屏幕的光映在脸上，室友们都睡了，宿舍里只有空调的嗡嗡声。

然后她点开评论框。

光标闪烁。她打了几个字，又删掉。再打，再删。反反复复，像在跟自己较劲。

她退出编辑，往上翻，找到那条动态下面的点赞按钮。

点了一个赞。

拇指大小的爱心亮起来，混在十几个赞里面，不起眼，不突兀。就像她这个人，在他生活里存在过的痕迹。

夏知晴把手机扣在胸口，闭上了眼。""",
    },
    # 以下两个场景请 April 替换为自己的小说片段
    # {
    #     "name": "环境描写",
    #     "category": "环境与氛围",
    #     "text": """（替换为你的环境描写片段）""",
    # },
    # {
    #     "name": "对话与潜台词",
    #     "category": "对话与潜台词",
    #     "text": """（替换为你的对话场景片段）""",
    # },
]


# ─── 参与对比的模型列表 ────────────────────────────────────────
# 注释掉没有 API Key 的模型，脚本会自动跳过未配置的模型

COMPARE_MODELS = [
    "glm-4-flash",     # 智谱（免费）
    "glm-4",           # 智谱（效果更好）
    "deepseek-chat",   # DeepSeek-V3
    "moonshot-v1-8k",  # Kimi
    "MiniMax-M1",      # MiniMax
]

# 评测模型（固定用 GLM-4，确保评测标准统一）
EVAL_MODEL = "glm-4"
EVAL_ENV_KEY = "ZHIPU_API_KEY"


# ─── 核心逻辑 ──────────────────────────────────────────────────

def get_api_key(model_name: str) -> str:
    """根据模型名获取对应的 API Key"""
    config = MODEL_CONFIG.get(model_name, {})
    env_key = config.get("env_key", "ZHIPU_API_KEY")
    return os.environ.get(env_key, "")


def run_single_model(model_name: str, input_text: str, category_hint: str) -> dict:
    """用单个模型进行改写"""
    config = MODEL_CONFIG.get(model_name, {})
    label = config.get("label", model_name)
    api_key = get_api_key(model_name)

    if not api_key:
        return {"label": label, "model": model_name, "status": "skipped", "reason": f"未设置 {config.get('env_key', '')} 环境变量"}

    try:
        print(f"  🔄 {label} 改写中...")
        start_time = time.time()
        result = rewrite_to_august_style(
            input_text=input_text,
            api_key=api_key,
            model=model_name,
            category_hint=category_hint,
        )
        elapsed = time.time() - start_time
        print(f"  ✅ {label} 完成 ({elapsed:.1f}s)")
        return {
            "label": label,
            "model": model_name,
            "provider": config.get("provider", ""),
            "status": "success",
            "result": result,
            "elapsed": round(elapsed, 1),
        }
    except Exception as e:
        print(f"  ❌ {label} 失败: {e}")
        return {"label": label, "model": model_name, "status": "error", "reason": str(e)}


def evaluate_result(original_text: str, rewritten_text: str) -> dict:
    """用 GLM-4 评测改写结果（统一评测标准）"""
    eval_key = os.environ.get(EVAL_ENV_KEY, "")
    if not eval_key:
        return {"status": "skipped", "reason": "未设置 ZHIPU_API_KEY，无法评测"}

    try:
        print(f"  📊 评测中...")
        result = evaluate_text(
            original_text=original_text,
            rewritten_text=rewritten_text,
            api_key=eval_key,
            model=EVAL_MODEL,
        )
        print(f"  ✅ 评测完成")
        return {"status": "success", **result}
    except Exception as e:
        print(f"  ❌ 评测失败: {e}")
        return {"status": "error", "reason": str(e)}


def generate_report(scene: dict, all_results: list) -> str:
    """生成 Markdown 对比报告"""
    lines = []
    lines.append(f"# 多模型改写对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 测试场景：{scene['name']}（{scene['category']}）")
    lines.append(f"> 评测模型：{EVAL_MODEL}（统一标准）")
    lines.append(f"> 参与模型：{len(all_results)} 个\n")
    lines.append("---\n")

    # ─── 评分总览表 ───
    lines.append("## 一、评分总览\n")
    lines.append("| 模型 | 提供商 | 状态 | 综合评分 | 具象精准 | 情感间接 | 对话潜台词 | 克制留白 | 环境映射 | 叙述声音 | 幽默反差 | 节奏控制 |")
    lines.append("|------|--------|------|---------|---------|---------|-----------|---------|---------|---------|---------|---------|")

    for r in all_results:
        if r["rewrite"]["status"] != "success":
            lines.append(f"| {r['rewrite']['label']} | {r['rewrite'].get('provider', '')} | ❌ {r['rewrite']['status']} | - | - | - | - | - | - | - | - | - |")
            continue

        eval_data = r.get("evaluation", {})
        if eval_data.get("status") != "success":
            lines.append(f"| {r['rewrite']['label']} | {r['rewrite'].get('provider', '')} | ✅改写 ❌评测 | - | - | - | - | - | - | - | - | - |")
            continue

        scores = {s["dimension"]: s["score"] for s in eval_data.get("scores", [])}
        overall = eval_data.get("overall_score", 0)
        lines.append(
            f"| {r['rewrite']['label']} | {r['rewrite'].get('provider', '')} | ✅ | "
            f"**{overall:.1f}** | "
            f"{scores.get('具象精准度', '-')} | "
            f"{scores.get('情感间接性', '-')} | "
            f"{scores.get('对话潜台词密度', '-')} | "
            f"{scores.get('克制留白程度', '-')} | "
            f"{scores.get('环境映射质量', '-')} | "
            f"{scores.get('叙述者声音', '-')} | "
            f"{scores.get('幽默反差运用', '-')} | "
            f"{scores.get('节奏控制', '-')} |"
        )

    lines.append("")

    # ─── 各模型改写结果 ───
    lines.append("## 二、各模型改写结果\n")

    for i, r in enumerate(all_results, 1):
        rw = r["rewrite"]
        lines.append(f"### {i}. {rw['label']}（{rw.get('provider', '')}）\n")

        if rw["status"] != "success":
            lines.append(f"> ⚠️ {rw.get('reason', '未知错误')}\n")
            continue

        if rw.get("elapsed"):
            lines.append(f"> 耗时：{rw['elapsed']}s\n")

        lines.append("**改写结果：**\n")
        lines.append(f"> {rw['result']}\n")

        # 评测详情
        ev = r.get("evaluation", {})
        if ev.get("status") == "success":
            lines.append("**评测详情：**\n")
            for s in ev.get("scores", []):
                score = s["score"]
                if score == 0 and ("无对话" in s.get("reason", "") or "不适合" in s.get("reason", "")):
                    lines.append(f"- ⚪ **{s['dimension']}**: N/A — {s['reason']}")
                else:
                    bar = "█" * score + "░" * (5 - score)
                    lines.append(f"- {'⭐' if score >= 4 else '🔹' if score >= 3 else '🔸'} **{s['dimension']}**: {score}/5 `{bar}` — {s['reason']}")
            lines.append(f"\n💬 _{ev.get('overall_comment', '')}_\n")

            suggestions = ev.get("improvement_suggestions", [])
            if suggestions:
                lines.append("**改进建议：**")
                for j, sugg in enumerate(suggestions, 1):
                    lines.append(f"  {j}. {sugg}")
                lines.append("")
        elif ev.get("status") == "error":
            lines.append(f"> ⚠️ 评测失败：{ev.get('reason', '')}\n")

        lines.append("---\n")

    # ─── 横向对比分析 ───
    lines.append("## 三、横向对比分析\n")
    lines.append("*(此部分由作者/分析师基于上方数据填写)*\n")
    lines.append("### 3.1 各模型风格倾向\n")
    lines.append("| 模型 | 风格倾向 | 强项 | 短板 |")
    lines.append("|------|---------|------|------|")
    lines.append("| | | | |\n")
    lines.append("### 3.2 共性问题\n")
    lines.append("- \n")
    lines.append("### 3.3 改进方向\n")
    lines.append("- \n")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  多模型改写对比脚本")
    print("=" * 60)

    # 检查可用的模型
    available_models = []
    for m in COMPARE_MODELS:
        key = get_api_key(m)
        config = MODEL_CONFIG.get(m, {})
        label = config.get("label", m)
        if key:
            available_models.append(m)
            print(f"  ✅ {label} — 已配置")
        else:
            print(f"  ⏭️ {label} — 未配置 {config.get('env_key', '')}，将跳过")

    if not available_models:
        print("\n❌ 没有可用的模型！请至少设置一个 API Key。")
        print("   详见 docs/API注册指南.md")
        return

    # 检查评测模型
    eval_key = os.environ.get(EVAL_ENV_KEY, "")
    if eval_key:
        print(f"\n  📊 评测模型：{EVAL_MODEL} — 已配置")
    else:
        print(f"\n  ⚠️ 评测模型 {EVAL_MODEL} 未配置 ZHIPU_API_KEY，将跳过评测")

    print(f"\n  共 {len(TEST_SCENES)} 个测试场景，{len(available_models)} 个模型\n")

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    all_scene_results = []

    for scene_idx, scene in enumerate(TEST_SCENES, 1):
        print(f"\n{'─' * 60}")
        print(f"  场景 {scene_idx}/{len(TEST_SCENES)}: {scene['name']}（{scene['category']}）")
        print(f"{'─' * 60}")

        # 生成 prompt（仅展示一次）
        if scene_idx == 1:
            prompt = get_prompt_only(
                input_text=scene["text"],
                category_hint=scene["category"],
            )
            print(f"\n  📋 Prompt 已生成（{len(prompt)} 字符）")
            print(f"  （完整 Prompt 见输出文件）\n")

        # 对每个模型运行改写
        scene_results = []
        for model_name in available_models:
            print(f"\n  ── 模型: {MODEL_CONFIG[model_name]['label']} ──")

            # 改写
            rewrite_result = run_single_model(model_name, scene["text"], scene["category"])

            # 评测（仅改写成功的才评测）
            eval_result = {"status": "skipped"}
            if rewrite_result["status"] == "success" and eval_key:
                eval_result = evaluate_result(scene["text"], rewrite_result["result"])

            scene_results.append({
                "rewrite": rewrite_result,
                "evaluation": eval_result,
            })

        # 生成报告
        report = generate_report(scene, scene_results)
        report_path = os.path.join(output_dir, f"对比报告_{scene['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  📄 报告已保存: {report_path}")

        # 保存 Prompt
        if scene_idx == 1:
            prompt_path = os.path.join(output_dir, f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"  📄 Prompt 已保存: {prompt_path}")

        # 保存 JSON 原始数据
        json_data = {
            "scene": scene,
            "timestamp": datetime.now().isoformat(),
            "eval_model": EVAL_MODEL,
            "results": scene_results,
        }
        json_path = os.path.join(output_dir, f"原始数据_{scene['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  📄 原始数据已保存: {json_path}")

        all_scene_results.extend(scene_results)

    # 最终摘要
    print(f"\n{'=' * 60}")
    print(f"  全部完成！")
    print(f"{'=' * 60}")
    success_count = sum(1 for r in all_scene_results if r["rewrite"]["status"] == "success")
    eval_count = sum(1 for r in all_scene_results if r.get("evaluation", {}).get("status") == "success")
    print(f"  改写成功: {success_count}/{len(all_scene_results)}")
    print(f"  评测成功: {eval_count}/{len(all_scene_results)}")
    print(f"  报告位置: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
