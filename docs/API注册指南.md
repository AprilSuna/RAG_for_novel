# API 注册指南

本文档介绍如何注册并获取各主流大模型的 API Key，用于多模型改写对比测试。

> 所有模型均兼容 OpenAI SDK，注册后拿到 API Key 即可使用。

---

## 必须配置

### 智谱 GLM（评测模型 + 改写模型）

- **注册地址**：https://open.bigmodel.cn/
- **步骤**：注册账号 → 控制台 → API Keys → 创建 Key
- **费用**：glm-4-flash 免费；glm-4 有免费额度
- **环境变量**：`ZHIPU_API_KEY`
- **用途**：改写（glm-4-flash / glm-4）+ 评测（glm-4，统一标准）

> ✅ 你已经注册过了，跳过这步。

---

## 可选配置（至少配 1-2 个用于对比）

### DeepSeek（强烈推荐 — 你要投的公司）

- **注册地址**：https://platform.deepseek.com/
- **步骤**：注册账号 → API Keys → 创建 Key
- **费用**：约 1元/百万输入token，2元/百万输出token（非常便宜，跑10次对比不到1毛钱）
- **模型名**：`deepseek-chat`（即 DeepSeek-V3）
- **环境变量**：`DEEPSEEK_API_KEY`

> 🔑 战略价值：你投的是 DeepSeek，面试时说"我测了 DeepSeek-V3 的文学改写能力，发现它在 XX 方面表现好/差"，直接命中岗位要求第一条。

### 通义千问 Qwen（阿里）

- **注册地址**：https://bailian.console.aliyun.com/
- **步骤**：用阿里云/支付宝账号登录 → API-KEY 管理 → 创建 Key
- **费用**：新用户有免费额度；qwen-plus 价格很低
- **模型名**：`qwen-plus`
- **环境变量**：`DASHSCOPE_API_KEY`

> 注意：创建 Key 后，到「模型广场」找到 qwen-plus，点进去打开「免费额度用完即停」，防止意外扣费。

### 月之暗面 Kimi

- **注册地址**：https://platform.moonshot.cn/
- **步骤**：注册账号 → API Key 管理 → 创建 Key
- **费用**：新用户有免费额度
- **模型名**：`moonshot-v1-8k`
- **环境变量**：`MOONSHOT_API_KEY`

---

## 环境变量设置

拿到 API Key 后，在终端中设置（每次开新终端都要设）：

### macOS / Linux

```bash
export ZHIPU_API_KEY=你的智谱key
export DEEPSEEK_API_KEY=你的deepseek_key
export DASHSCOPE_API_KEY=你的阿里key
export MOONSHOT_API_KEY=你的kimi_key
export MINIMAX_API_KEY=你的minimax_key
```

### Windows (PowerShell)

```powershell
$env:ZHIPU_API_KEY="你的智谱key"
$env:DEEPSEEK_API_KEY="你的deepseek_key"
$env:DASHSCOPE_API_KEY="你的阿里key"
$env:MOONSHOT_API_KEY="你的kimi_key"
```

### 永久设置（推荐）

把上面的 export 命令加到 `~/.zshrc`（macOS）或 `~/.bashrc` 中，以后就不用每次都设了。

---

## 运行对比

设置好环境变量后，在项目目录下运行：

```bash
cd /Users/guosihong/Desktop/august_eight_rag
python compare_models.py
```

脚本会自动：
1. 检测哪些模型已配置（没配的自动跳过）
2. 对每个测试场景，用所有已配置模型进行改写
3. 用 GLM-4 统一评测所有改写结果（8维度打分）
4. 生成 Markdown 对比报告 + JSON 原始数据到 `output/` 目录

---

## 测试场景

脚本中预设了「心理描写」场景（序章 QQ 空间片段）。
如需添加更多场景，编辑 `compare_models.py` 中的 `TEST_SCENES` 列表：

```python
TEST_SCENES = [
    {
        "name": "心理描写",
        "category": "心理与情感",
        "text": """（你的测试文本）""",
    },
    {
        "name": "环境描写",
        "category": "环境与氛围",
        "text": """（你的测试文本）""",
    },
    {
        "name": "对话与潜台词",
        "category": "对话与潜台词",
        "text": """（你的测试文本）""",
    },
]
```

建议每种场景选一段 200-400 字的片段，覆盖不同的写作难度。
