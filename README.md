# TTSDataGen

> 本地优先的 TTS 训练对话数据生成工具，支持 RAG 检索、多阶段优化与可验证的 A/B 对话输出

<div align="center">

![Version](https://img.shields.io/badge/version-0.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-green.svg)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)

**创作者**: [Micheli V](https://micheliliuv87.github.io/)

[效果演示](#-效果演示) ｜ [功能特性](#-功能特性) ｜ [安装](#-安装) ｜ [使用指南](#-使用指南) ｜ [架构文档](ARCHITECTURE.md) ｜ [常见问题](#-常见问题)

<p align="center">
  中文 | <a href="README_en.md">English</a>
</p>

</div>

---

##  🎬 效果演示

<div align="center">
  <video src="https://github.com/user-attachments/assets/1c32b448-a82d-4cab-9f72-f4dbca109085"
         width="800"
         controls
         loop
         muted>
    您的浏览器不支持视频播放
  </video>
  <p><em>AI 生成多轮次A/B对话 文字转语音模型训练数据</em></p>
</div>

---

## 📖 简介

TTSDataGen V0.2 是一个面向 Text-to-Speech 训练数据的本地化对话生成工具，能够根据中文或英文需求，生成自然、可读、可扩展的多轮中文 A/B 对话。

它能够：

- 💬 **生成多轮 A/B 对话**，适用于 TTS 训练数据构建
- 🔍 **使用本地 RAG 检索**，从知识库中提取相关内容辅助生成
- 🗣️ **优化中文对话风格**，避免百科总结式输出，让 A/B 更像真实交流
- 🛠️ **执行多阶段优化**，支持批注、扩写、润色和最终校验
- ✅ **自动验证输出质量**，检查轮数、A/B 格式、内容密度和完整性
- 🧠 **本地模型驱动**，通过 LM Studio 调用 Qwen3-4B / Qwen3-32B

---

## ✨ 功能特性

### 🎯 核心能力

- 💬 **多轮 A/B 对话生成**  
  根据用户输入的主题或需求，生成结构清晰的中文 A/B 多轮对话，适用于 Text-to-Speech 训练数据构建。

- 🔍 **本地 RAG 内容检索**  
  从本地知识库中检索相关文本片段，为对话生成提供内容依据，减少完全凭空生成的问题。

- 🧩 **Source Anchor Pack 构建**  
  将检索结果整理成更紧凑、更可用的内容锚点，帮助模型优先使用高价值 source material。

- 📝 **Source-grounded 生成**  
  生成结果尽量保留原始材料中的具体细节、事件、例子和表达，而不是只输出泛泛总结。


### 🗣️ 对话风格

- 🧑‍🤝‍🧑 **自然的双人交流**  
  输出以 A/B 双人对话形式展开，强调有来有回的交流感，而不是单向讲解。

- 🗣️ **更适合中文朗读**  
  通过扩写和润色阶段改善中文表达，减少翻译腔、重复开头和生硬句式。

- 🌱 **支持内容扩展**  
  对过短、过薄或信息不足的句子进行扩写，让对话更完整、更适合 TTS 训练。

- 🚫 **避免百科式输出**  
  尽量避免把对话写成说明文或资料摘要，而是保持口语化、场景化和可朗读性。

### 🔁 Pipeline 流程

- ⚡ **Draft Mode**  
  快速完成检索、Source Anchor Pack 构建、初稿生成和基础校验，适合快速预览结果。

- 🧪 **Full Mode**  
  执行完整流程：检索 → Source Anchor Pack → 初稿生成 → 校验 → 批注 → 扩写 → 再校验 → 润色 → 最终校验。

- ✅ **多阶段质量检查**  
  在生成、扩写和润色后分别进行 validation，尽量避免格式错误、轮数错误或内容密度不足的问题。

### 🛠️ 技术亮点

- 🖥️ **Local-first 架构**  
  项目优先在本地运行，尽量减少外部 API 成本和数据依赖。

- 🧠 **LM Studio 本地模型调用**  
  通过 OpenAI-compatible API 调用本地大模型，当前主要使用 Qwen3-4B 和 Qwen3-32B。

- 🗄️ **本地向量数据库**  
  使用本地 vector database 进行语义检索，支持基于已有文本资料的 RAG 生成。

- 📐 **规则驱动生成行为**  
  使用独立 rule files 控制 generation、critique、expansion、polish 和 validation，不需要频繁修改 Python 代码。

- ⚙️ **配置化 Prompt 与 Pipeline**  
  Prompt templates 和模型参数放在配置文件中，方便调整不同阶段的生成行为。

- 📁 **完整元数据记录**  
  保存 Prompt JSON、Validation JSON、Metadata JSON 和 Pipeline run JSON，方便复查、调试和迭代。

---

## 🚀 安装

TTSDataGen V0.2 采用本地优先架构运行。安装前请确保你已经安装：

- Python 3.12+
- Git
- LM Studio
- Qwen3-4B / Qwen3-32B 本地模型
- 已准备好的本地 RAG 数据库，或可自行重新构建向量库

---

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/TTSDataGen.git
cd TTSDataGen
```

### 2. 创建 Python 环境

推荐使用`conda`:
```bash
conda create -n ttsdatagen python=3.12 -y
conda activate ttsdatagen
```

### 3. 安装依赖
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 启动LM Studio
打开 LM Studio，并加载本地模型：
```bash
Generator / Critic / Expander / Polisher:
qwen3-32b-mlx

Query Rewrite:
qwen3-4b
```
请确保 LM Studio 的本地 API Server 已启动：
```
http://localhost:1234/v1
```
如果 LM Studio 没有启动，pipeline 将无法调用本地大模型。

### 5. 准备本地向量数据库

如果你已经有了准备好的本地 RAG 数据库，可以将其路径配置到 (注意修改任何有关HappyScribe的路径配置)：

```bash
vector_db/chroma_content/
```

如果你需要重新构建 RAG 数据库，可以运行：

```bash
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

如果你没有数据，可以使用以下脚本从 HappyScribe 上抓取公开的字幕并跑上面的构建 RAG 数据库脚本：（这组数据爬虫约10个小时，构建 RAG 数据库约17个小时，建议提前配置电脑或服务器）

```bash 
bash scripts/run_happyscribe_scrape.sh
bash scripts/run_happyscribe_audit.sh
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

向量库构建完成后，系统会在本地使用 ChromaDB 进行内容检索。

### 6. 运行 Pipeline

```bash
bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是早餐文化、家庭习惯和代际差异"
```

运行快速草稿模式：
```bash
PIPELINE_MODE=draft bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是火车旅行、陌生人和人生选择"
```


### 8. 查看输出结果

最终生成结果通常保存在：

```bash
outputs/polishes/
```

常见输出文件包括：

```
polished_expanded_dialogue_pipeline_*.md
polished_expanded_dialogue_pipeline_*.meta.json
polished_expanded_dialogue_pipeline_*.prompt.json
polished_expanded_dialogue_pipeline_*.validation.json
```

其中 .md 文件是最终对话文本，.json 文件用于记录 prompt、metadata、validation 和 pipeline 运行状态。

---

## 📚 使用指南

TTSDataGen 支持两种使用方式：推荐使用 Streamlit UI，也可以通过命令行直接运行 pipeline。


### 1. 使用 Streamlit UI

启动 LM Studio 并加载本地模型后，运行：

```bash
streamlit run app/streamlit_app.py
```

打开页面后，按照界面提示输入生成需求，例如：

```bash
生成30轮A与B对话，主题是早餐文化、家庭习惯和代际差异，风格自然，适合中文朗读。
```

在 UI 中可以选择：

* Draft Mode：快速生成初稿，适合预览效果
* Full Mode：执行完整流程，适合生成最终结果

Full Mode 会运行：

检索 → Source Anchor Pack → 初稿生成 → 校验 → 批注 → 扩写 → 再校验 → 润色 → 最终校验

### 2. 使用命令行运行

推荐使用清晰的主题描述：

```bash
生成30轮A与B对话，主题是[主题1]、[主题2]和[主题3]，风格自然，适合中文朗读。
```

也可以加入更具体的风格要求：
```bash
生成30轮A与B对话，主题是城市生活、孤独感和咖啡馆，风格像两个朋友聊天，不要写成百科解释。
```

主要查看：

```bash
polished_expanded_dialogue_pipeline_*.md
```
其他 JSON 文件用于记录 prompt、metadata、validation 和 pipeline 运行状态：
```bash
*.meta.json
*.prompt.json
*.validation.json
outputs/pipeline_runs/pipeline_*.json
```

---

---

## ❓ 常见问题

### Q: 是否必须使用外部 API？

**A**: 不必须。  
TTSDataGen V0.2 采用 local-first 架构，主要通过 LM Studio 在本地调用模型。正常运行时不需要 OpenAI、Google、Kling 或其他第三方生成 API。

你需要准备的是：

- 本地安装 LM Studio
- 在 LM Studio 中加载 Qwen3-4B / Qwen3-32B
- 启动 LM Studio 的本地 API Server
- 准备好本地 RAG 向量数据库

默认本地 API 地址为：

```text
http://localhost:1234/v1
```

---

### Q: 为什么必须启动 LM Studio？

**A**: 因为 pipeline 的多个阶段都需要调用本地大模型。

其中：

```text
Qwen3-4B
  → Query Rewrite

Qwen3-32B
  → Dialogue Generation
  → Critique
  → Expansion
  → Polish
```

如果 LM Studio 没有启动，或者模型名称和配置文件中的 `model` 字段不一致，pipeline 会无法继续运行。

---

### Q: Draft Mode 和 Full Mode 有什么区别？

**A**:

| 模式 | 适合场景 | 执行流程 |
|---|---|---|
| Draft Mode | 快速预览结果 | 检索 → Source Anchor Pack → 初稿生成 → 基础校验 |
| Full Mode | 生成最终可用结果 | 检索 → Source Anchor Pack → 初稿生成 → 校验 → 批注 → 扩写 → 再校验 → 润色 → 最终校验 |

Draft Mode 更快，适合测试主题是否能检索到合适素材。  
Full Mode 更慢，但输出质量通常更高，更适合生成最终 TTS 训练文本。

---

### Q: Full Mode 为什么运行很慢？

**A**: Full Mode 会多次调用本地大模型，并执行多个阶段：

```text
generate
validate
critique
expand
validate
polish
validate
```

其中 expansion 和 polish 可能会分 batch 调用模型，因此运行时间取决于：

- 本地机器性能
- 模型大小
- 对话轮数
- source anchor 数量
- 需要扩写的 speaker lines 数量

在本地 32B 模型下，30 轮 Full Mode 可能需要较长时间，这是正常现象。

---

### Q: 最终结果在哪里？

**A**: 最终结果通常保存在：

```text
outputs/polishes/
```

主要查看：

```text
polished_expanded_dialogue_pipeline_*.md
```

常见伴随文件包括：

```text
*.meta.json
*.prompt.json
*.validation.json
outputs/pipeline_runs/pipeline_*.json
```

其中：

- `.md` 是最终对话文本
- `.meta.json` 记录运行元数据
- `.prompt.json` 记录实际发送给模型的 prompt
- `.validation.json` 记录校验结果
- `pipeline_*.json` 记录完整 pipeline 状态

---

### Q: 没有本地 RAG 数据库可以运行吗？

**A**: 当前版本默认依赖本地 RAG 数据库。  
如果没有可用的：

```text
vector_db/chroma_content/
```

需要先构建向量库。

可以运行：

```bash
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

如果你还没有原始数据，可以先运行 HappyScribe 抓取脚本，再构建 RAG 数据库：

```bash
bash scripts/run_happyscribe_scrape.sh
bash scripts/run_happyscribe_audit.sh
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

注意：完整抓取和向量库构建可能需要很长时间。

---

### Q: 生成结果太短怎么办？

**A**: 建议使用 Full Mode。  
Full Mode 会在初稿之后执行 critique、expansion 和 polish，其中 expansion 阶段会重点扩写过短、过薄或信息不足的 speaker lines。

如果仍然觉得内容太短，可以尝试：

1. 减少一次生成的轮数，先测试主题质量
2. 使用更具体的主题描述
3. 确认 RAG 检索结果是否与主题相关
4. 检查 `outputs/pipeline_runs/pipeline_*.json` 中的 source quality summary
5. 查看 `.validation.json` 中是否存在 `high_short_line_ratio` 或 `low_developed_line_ratio`

---

### Q: 生成结果偏题怎么办？

**A**: 可以先检查 RAG 检索阶段。

建议查看：

```text
outputs/source_packs/latest_source_pack.json
outputs/source_packs/latest_source_anchor_pack.json
```

重点检查：

- query rewrite 是否正确理解了主题
- retrieved sources 是否和主题相关
- source anchors 是否过少
- source anchors 是否被大量 rejected
- coverage 是否为 weak 或 none

如果检索结果本身偏题，后续生成质量通常也会受到影响。

---

### Q: 可以生成英文对话吗？

**A**: 当前项目主要围绕中文 A/B 对话优化，尤其是中文 TTS 训练文本。  
虽然部分配置中支持 `language` 字段，但当前规则、扩写和润色策略主要针对中文输出设计。

如果需要稳定生成英文对话，建议后续单独配置英文 prompt templates、rules 和 validation thresholds。

---

### Q: 可以商用吗？

**A**: 请以本仓库的许可证说明为准。  
当前 README badge 标注为：

```text
All Rights Reserved
```

这意味着未经作者明确许可，不应直接复制、分发、改造或商用本项目代码。

如果你计划开放源代码或允许他人使用，可以将许可证改为 MIT、Apache-2.0 或其他开源许可证，并同步更新 README badge 和 `LICENSE` 文件。

---

## 🛡️ 安全与隐私说明

TTSDataGen V0.2 采用 local-first 架构，默认尽量在本地完成模型调用、向量检索和文件生成。

### 本地模型调用

项目默认通过 LM Studio 的本地 OpenAI-compatible API 调用模型：

```text
http://localhost:1234/v1
```

正常情况下，用户输入、检索结果和生成内容都在本地机器上处理，不需要发送到外部云端模型 API。

---

### 本地数据与输出

以下目录可能包含大量本地数据、生成结果、日志或中间产物：

```text
data/
vector_db/
outputs/
logs/
```

其中可能包含：

- 原始 transcript 数据
- 清洗后的文本
- RAG chunks
- Chroma 向量数据库
- 用户输入 query
- 生成的 dialogue markdown
- prompt JSON
- validation JSON
- pipeline metadata
- 运行日志

如果你的数据中包含敏感内容，请不要将这些目录提交到 GitHub。

---

### `.gitignore` 建议

建议确保以下目录不会被提交：

```gitignore
# Logs
logs/

# Large local data
data/raw/
data/interim/
data/processed/

# Runtime outputs
outputs/

# Local vector database
vector_db/

# Env variables / tokens
.env
.env.*

# Model / HF caches
.cache/
huggingface_cache/
models/
```

---

### Prompt 与日志安全

TTSDataGen 会保存 prompt JSON 和 pipeline metadata，方便调试和复查。

这些文件通常位于：

```text
outputs/dialogues/
outputs/expansions/
outputs/polishes/
outputs/pipeline_runs/
```

如果用户输入或本地 source material 中包含敏感信息，prompt JSON 和 metadata 中也可能间接包含这些内容。  
在公开分享日志或输出文件前，请先检查其中是否包含不应公开的文本。

---

### 提交前检查

提交代码前，建议检查是否误提交了大文件、日志或本地数据：

```bash
git status
```

也可以检查是否存在潜在密钥或敏感配置：

```bash
grep -r "api_key\|secret\|token\|password" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=vector_db \
  --exclude-dir=outputs \
  --exclude-dir=data
```

如果项目不使用外部 API，这个命令一般不应该返回真实密钥。

---

## 📝 更新日志

### v0.2.0

- 🖥️ **新增 Streamlit 本地 UI**
  - 支持用户输入生成需求
  - 支持 Draft / Full 模式选择
  - 支持查看生成结果、运行日志和校验状态
  - 支持下载最终 Markdown 输出

- 🔍 **完善本地 RAG 流程**
  - 支持 query rewrite
  - 支持 Chroma 本地向量检索
  - 支持 source pack 输出
  - 支持 Source Anchor Pack 构建

- 💬 **升级多阶段对话生成 pipeline**
  - 初稿生成
  - 结构校验
  - 质量批注
  - 内容扩写
  - 表达润色
  - 最终校验

- ✅ **新增确定性 validation**
  - 检查轮数
  - 检查 A/B 编号
  - 检查 speaker 交替
  - 检查 Source Appendix
  - 检查短句比例和充分展开比例
  - 检查 source / retrieval 泄露

- 📁 **完善运行产物记录**
  - 保存 prompt JSON
  - 保存 metadata JSON
  - 保存 validation JSON
  - 保存 pipeline run JSON
  - 保存运行日志

---

### v0.1.0

- ✨ 初始版本
- 🔍 支持本地 RAG 检索
- 🧠 支持 LM Studio 本地模型调用
- 💬 支持中文 A/B 对话生成
- 📦 支持基础 source pack 输出
- 🧪 支持命令行运行 pipeline

---

## 🤝 贡献指南

欢迎贡献、反馈问题或提出改进建议。

你可以从以下方向参与：

### 报告问题

如果你遇到 bug 或生成质量问题，可以在 GitHub Issues 中提交，并尽量包含：

- 操作系统和 Python 版本
- 使用的 LM Studio 模型名称
- 运行命令或 UI 操作步骤
- 错误信息
- 相关日志文件
- 对应的 `pipeline_*.json`
- 生成结果或 validation report

Issues 地址：

```text
https://github.com/Micheliliuv87/TTSDataGen/issues
```

如果你的仓库名称不同，请将上面的链接替换为实际 Issues 地址。

---

### 改进规则文件

项目的生成行为主要由以下规则文件控制：

```text
knowledge_base/rules/generation_rules.jsonl
knowledge_base/rules/critique_rules.jsonl
knowledge_base/rules/expand_rules.jsonl
knowledge_base/rules/polish_rules.jsonl
knowledge_base/rules/validation_rules.jsonl
```

如果你发现某类问题反复出现，可以考虑新增或调整对应阶段的 rule。

建议修改后测试：

```bash
bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是城市生活、孤独感和咖啡馆"
```

---

### 改进 Prompt 或配置

Prompt 模板和模型参数主要位于：

```text
configs/
```

尤其是：

```text
configs/prompt_templates.yaml
configs/generation.yaml
configs/critic.yaml
configs/expander.yaml
configs/polisher.yaml
configs/rag.yaml
```

修改配置后，建议保留对应的 `.prompt.json` 和 `.validation.json`，方便比较不同版本的生成效果。

---

### 改进 UI

Streamlit UI 位于：

```text
app/streamlit_app.py
```

可以改进的方向包括：

- 更清楚的运行进度展示
- 更好的错误提示
- 更方便的下载按钮
- 更清楚的 validation summary
- 更稳定的任务取消机制
- 更好的 Source Appendix 展示

---

## 📄 许可证

当前项目许可证为：


[LICENSE](LICENSE) <ins>**All Rights Reserved**</ins>


除非作者另行授权，否则不允许复制、分发、修改、再发布或用于商业用途。

如需使用、引用、二次开发或商业合作，请先联系作者获得明确许可。

---

## 🙏 致谢

感谢以下项目和工具为 TTSDataGen 提供支持：

- **LM Studio**：提供本地 OpenAI-compatible 模型服务
- **Qwen**：提供本地大语言模型能力
- **BAAI/bge-m3**：提供本地 embedding 模型
- **ChromaDB**：提供本地向量数据库
- **Streamlit**：提供本地交互式 UI
- **HappyScribe public transcript pages**：作为本地 RAG 数据构建的公开 transcript 来源
- **Python 开源生态**：提供数据处理、配置管理和 pipeline 构建基础

---

## 📞 联系方式

- **创作者**: [Micheli V](https://micheliliuv87.github.io/)
- **GitHub**: [@Micheliliuv87](https://github.com/Micheliliuv87)
- **Issues**: [GitHub Issues](https://github.com/Micheliliuv87/TTSDataGen/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎给一个 Star！**

Made with ❤️ by Micheli V

</div>