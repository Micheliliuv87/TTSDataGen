# TTSDataGen

> A local-first dialogue data generation tool for TTS training, with RAG retrieval, multi-stage refinement, and verifiable A/B dialogue output.

<div align="center">

![Version](https://img.shields.io/badge/version-0.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-green.svg)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)

**Creator**: [Micheli V](https://micheliliuv87.github.io/)

[Demo](#-demo) ｜ [Features](#-features) ｜ [Installation](#-installation) ｜ [Usage](#-usage) ｜ [Architecture](ARCHITECTURE_en.md) ｜ [FAQ](#-faq)

<p align="center">
  <a href="README.md">中文</a> | English
</p>

</div>

---

## 🎬 Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/1c32b448-a82d-4cab-9f72-f4dbca109085"
         width="800"
         controls
         loop
         muted>
    Your browser does not support video playback.
  </video>
  <p><em>AI-generated multi-round A/B dialogue data for text-to-speech model training</em></p>
</div>

---

## 📖 Introduction

TTSDataGen V0.2 is a local-first dialogue generation tool for Text-to-Speech training data. It can generate natural, readable, and extensible multi-round Chinese A/B dialogues from Chinese or English user requests.

It can:

- 💬 **Generate multi-round A/B dialogues** for TTS training data construction.
- 🔍 **Use local RAG retrieval** to retrieve relevant source material from a local knowledge base.
- 🗣️ **Optimize Chinese dialogue style** by avoiding encyclopedia-like summaries and making A/B exchanges feel more conversational.
- 🛠️ **Run multi-stage refinement**, including critique, expansion, polish, and final validation.
- ✅ **Automatically validate output quality**, including round count, A/B format, content density, and structural completeness.
- 🧠 **Run with local models** through LM Studio, currently using Qwen3-4B and Qwen3-32B.

---

## ✨ Features

### 🎯 Core Capabilities

- 💬 **Multi-round A/B dialogue generation**  
  Generate clearly structured Chinese A/B multi-round dialogues from a user topic or request, suitable for Text-to-Speech training data.

- 🔍 **Local RAG content retrieval**  
  Retrieve relevant text chunks from a local knowledge base to ground generation and reduce unsupported free-form output.

- 🧩 **Source Anchor Pack construction**  
  Convert retrieval results into compact and usable source anchors, helping the model focus on high-value source material.

- 📝 **Source-grounded generation**  
  The generated dialogue aims to preserve concrete details, events, examples, and expressions from the source material instead of producing only generic summaries.

### 🗣️ Dialogue Style

- 🧑‍🤝‍🧑 **Natural two-person conversation**  
  Output is structured as A/B dialogue, emphasizing back-and-forth exchange rather than one-way explanation.

- 🗣️ **Better suited for Chinese speech**  
  Expansion and polish stages improve Chinese fluency, reduce translation-like phrasing, repeated openings, and stiff sentence structures.

- 🌱 **Content expansion support**  
  Short, thin, or underdeveloped lines can be expanded to make the dialogue more complete and more suitable for TTS training.

- 🚫 **Avoids encyclopedia-style output**  
  The system tries to avoid turning dialogue into explanatory essays or data summaries, keeping it more oral, contextual, and readable aloud.

### 🔁 Pipeline Modes

- ⚡ **Draft Mode**  
  Quickly runs retrieval, Source Anchor Pack construction, initial generation, and basic validation. Suitable for previewing results.

- 🧪 **Full Mode**  
  Runs the full workflow: retrieval → Source Anchor Pack → initial generation → validation → critique → expansion → re-validation → polish → final validation.

- ✅ **Multi-stage quality checks**  
  Validation is run after generation, expansion, and polish to reduce formatting errors, round-count errors, and content-density issues.

### 🛠️ Technical Highlights

- 🖥️ **Local-first architecture**  
  The project is designed to run locally as much as possible, reducing external API cost and data dependency.

- 🧠 **Local model calls via LM Studio**  
  Uses LM Studio's OpenAI-compatible API to call local large language models. Current main models are Qwen3-4B and Qwen3-32B.

- 🗄️ **Local vector database**  
  Uses a local vector database for semantic retrieval and source-grounded RAG generation.

- 📐 **Rule-driven generation behavior**  
  Independent rule files control generation, critique, expansion, polish, and validation without requiring frequent Python code changes.

- ⚙️ **Configurable prompts and pipeline behavior**  
  Prompt templates and model parameters are stored in config files, making it easier to adjust behavior at different stages.

- 📁 **Complete metadata logging**  
  Saves Prompt JSON, Validation JSON, Metadata JSON, and Pipeline run JSON for review, debugging, and iteration.

---

## 🚀 Installation

TTSDataGen V0.2 uses a local-first runtime architecture. Before installation, make sure you have:

- Python 3.12+
- Git
- LM Studio
- Local Qwen3-4B / Qwen3-32B models
- A prepared local RAG database, or source data from which one can be rebuilt

---

### 1. Clone the repository

```bash
git clone https://github.com/Micheliliuv87/TTSDataGen.git
cd TTSDataGen
```

### 2. Create a Python environment

Using `conda` is recommended:

```bash
conda create -n ttsdatagen python=3.12 -y
conda activate ttsdatagen
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start LM Studio

Open LM Studio and load the local models:

```text
Generator / Critic / Expander / Polisher:
qwen3-32b-mlx

Query Rewrite:
qwen3-4b
```

Make sure the LM Studio local API server is running:

```text
http://localhost:1234/v1
```

If LM Studio is not running, the pipeline cannot call the local models.

### 5. Prepare the local vector database

If you already have a prepared local RAG database, configure or place it at the expected path. Also make sure to update any HappyScribe-related paths if your data source is different:

```text
vector_db/chroma_content/
```

To rebuild the RAG database from processed data, run:

```bash
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

If you do not have local data yet, you can first scrape public transcripts from HappyScribe and then build the RAG database. This full process can take a long time: scraping may take about 10 hours, and building the RAG database may take about 17 hours, depending on your machine and dataset size.

```bash
bash scripts/run_happyscribe_scrape.sh
bash scripts/run_happyscribe_audit.sh
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

After the vector database is built, the system will use ChromaDB locally for content retrieval.

### 6. Run the pipeline

```bash
bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是早餐文化、家庭习惯和代际差异"
```

Run Draft Mode:

```bash
PIPELINE_MODE=draft bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是火车旅行、陌生人和人生选择"
```

### 7. View outputs

Final outputs are usually saved under:

```text
outputs/polishes/
```

Common output files include:

```text
polished_expanded_dialogue_pipeline_*.md
polished_expanded_dialogue_pipeline_*.meta.json
polished_expanded_dialogue_pipeline_*.prompt.json
polished_expanded_dialogue_pipeline_*.validation.json
```

The `.md` file is the final dialogue text. The `.json` files record prompts, metadata, validation results, and pipeline runtime state.

---

## 📚 Usage

TTSDataGen supports two usage modes: the recommended Streamlit UI and direct CLI execution.

### 1. Use the Streamlit UI

After starting LM Studio and loading the local models, run:

```bash
streamlit run app/streamlit_app.py
```

In the UI, enter a generation request such as:

```text
生成30轮A与B对话，主题是早餐文化、家庭习惯和代际差异，风格自然，适合中文朗读。
```

The UI supports:

- **Draft Mode**: quickly generate a draft for preview.
- **Full Mode**: run the complete workflow for final output.

Full Mode runs:

```text
Retrieval → Source Anchor Pack → Initial Generation → Validation → Critique → Expansion → Re-validation → Polish → Final Validation
```

### 2. Use the CLI

Use a clear topic description:

```text
生成30轮A与B对话，主题是[主题1]、[主题2]和[主题3]，风格自然，适合中文朗读。
```

You can also add more specific style requirements:

```text
生成30轮A与B对话，主题是城市生活、孤独感和咖啡馆，风格像两个朋友聊天，不要写成百科解释。
```

Main output to inspect:

```text
polished_expanded_dialogue_pipeline_*.md
```

Other JSON files record prompts, metadata, validation reports, and pipeline run state:

```text
*.meta.json
*.prompt.json
*.validation.json
outputs/pipeline_runs/pipeline_*.json
```

---

## ❓ FAQ

### Q: Do I need external APIs?

**A**: No.  
TTSDataGen V0.2 uses a local-first architecture and primarily calls local models through LM Studio. In normal usage, it does not require OpenAI, Google, Kling, or other third-party generation APIs.

You need:

- LM Studio installed locally
- Qwen3-4B / Qwen3-32B loaded in LM Studio
- The LM Studio local API server running
- A prepared local RAG vector database

Default local API address:

```text
http://localhost:1234/v1
```

---

### Q: Why must LM Studio be running?

**A**: Multiple pipeline stages call local large language models.

```text
Qwen3-4B
  → Query Rewrite

Qwen3-32B
  → Dialogue Generation
  → Critique
  → Expansion
  → Polish
```

If LM Studio is not running, or if the model name does not match the `model` field in the config files, the pipeline cannot continue.

---

### Q: What is the difference between Draft Mode and Full Mode?

**A**:

| Mode | Best for | Workflow |
|---|---|---|
| Draft Mode | Fast preview | Retrieval → Source Anchor Pack → Initial Generation → Basic Validation |
| Full Mode | Final usable output | Retrieval → Source Anchor Pack → Initial Generation → Validation → Critique → Expansion → Re-validation → Polish → Final Validation |

Draft Mode is faster and useful for testing whether a topic retrieves suitable source material.  
Full Mode is slower but usually produces higher-quality output, making it better for final TTS training text.

---

### Q: Why is Full Mode slow?

**A**: Full Mode calls local models multiple times and runs several stages:

```text
generate
validate
critique
expand
validate
polish
validate
```

Expansion and polish may call the model in batches, so runtime depends on:

- local hardware performance
- model size
- number of dialogue rounds
- number of source anchors
- number of speaker lines that need expansion

With a local 32B model, a 30-round Full Mode run may take a long time. This is expected.

---

### Q: Where are the final outputs?

**A**: Final outputs are usually saved under:

```text
outputs/polishes/
```

Main file to inspect:

```text
polished_expanded_dialogue_pipeline_*.md
```

Common companion files include:

```text
*.meta.json
*.prompt.json
*.validation.json
outputs/pipeline_runs/pipeline_*.json
```

- `.md` is the final dialogue text.
- `.meta.json` records runtime metadata.
- `.prompt.json` records the actual prompt sent to the model.
- `.validation.json` records validation results.
- `pipeline_*.json` records full pipeline state.

---

### Q: Can I run it without a local RAG database?

**A**: The current version assumes a local RAG database by default.  
If this path is not available:

```text
vector_db/chroma_content/
```

you need to build the vector database first.

```bash
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

If you do not have raw data yet, first run the HappyScribe scraping workflow, then build the RAG database:

```bash
bash scripts/run_happyscribe_scrape.sh
bash scripts/run_happyscribe_audit.sh
bash scripts/run_clean_sources.sh
bash scripts/run_chunk_sources.sh
bash scripts/run_build_rag.sh
```

The full scraping and vector database build process can take a long time.

---

### Q: What should I do if the generated dialogue is too short?

**A**: Use Full Mode.  
Full Mode runs critique, expansion, and polish after the initial draft. The expansion stage focuses on expanding speaker lines that are too short, too thin, or underdeveloped.

You can also try:

1. Generate fewer rounds first to test topic quality.
2. Use a more specific topic description.
3. Check whether the RAG retrieval results are relevant.
4. Inspect the source quality summary in `outputs/pipeline_runs/pipeline_*.json`.
5. Check whether `.validation.json` contains `high_short_line_ratio` or `low_developed_line_ratio`.

---

### Q: What should I do if the output drifts off topic?

**A**: Check the RAG retrieval stage first.

Useful files:

```text
outputs/source_packs/latest_source_pack.json
outputs/source_packs/latest_source_anchor_pack.json
```

Check whether:

- query rewrite understood the topic correctly
- retrieved sources are relevant
- source anchors are too few
- too many source anchors were rejected
- coverage is `weak` or `none`

If retrieval is off-topic, later generation quality will usually suffer.

---

### Q: Can it generate English dialogue?

**A**: The current project is mainly optimized for Chinese A/B dialogue, especially Chinese TTS training text.  
Some configs support a `language` field, but the current rules, expansion strategy, and polish strategy are primarily designed for Chinese output.

For stable English dialogue generation, it is better to add separate English prompt templates, rules, and validation thresholds.

---

### Q: Can I use it commercially?

**A**: Please follow the repository license.  
The current README badge indicates:

```text
All Rights Reserved
```

This means the project code should not be copied, redistributed, modified, republished, or used commercially without explicit permission from the author.

If you plan to open-source the project or allow broader use, update the license to MIT, Apache-2.0, or another license and synchronize the README badge and `LICENSE` file.

---

## 🛡️ Security and Privacy

TTSDataGen V0.2 uses a local-first architecture and tries to keep model calls, vector retrieval, and file generation on the local machine by default.

### Local model calls

The project calls models through LM Studio's local OpenAI-compatible API:

```text
http://localhost:1234/v1
```

In normal usage, user input, retrieval results, and generated content are processed locally and do not need to be sent to external cloud model APIs.

---

### Local data and outputs

The following directories may contain large local datasets, generated outputs, logs, or intermediate artifacts:

```text
data/
vector_db/
outputs/
logs/
```

They may contain:

- raw transcript data
- cleaned text
- RAG chunks
- Chroma vector database
- user queries
- generated dialogue markdown
- prompt JSON
- validation JSON
- pipeline metadata
- runtime logs

If your data contains sensitive content, do not commit these directories to GitHub.

---

### Recommended `.gitignore`

Make sure the following directories and files are not committed:

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

### Prompt and log safety

TTSDataGen saves prompt JSON and pipeline metadata for debugging and review.

These files are usually stored under:

```text
outputs/dialogues/
outputs/expansions/
outputs/polishes/
outputs/pipeline_runs/
```

If user input or local source material contains sensitive information, prompt JSON and metadata may also indirectly contain sensitive text.  
Before sharing logs or output files publicly, review them first.

---

### Pre-commit checks

Before committing, check for accidentally committed large files, logs, or local data:

```bash
git status
```

You can also scan for potential secrets or sensitive config values:

```bash
grep -r "api_key\|secret\|token\|password" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=vector_db \
  --exclude-dir=outputs \
  --exclude-dir=data
```

If the project does not use external APIs, this command generally should not return real secrets.

---

## 📝 Changelog

### v0.2.0

- 🖥️ **Added local Streamlit UI**
  - User generation request input
  - Draft / Full mode selection
  - Generated result, runtime log, and validation status display
  - Final Markdown download support

- 🔍 **Improved local RAG workflow**
  - Query rewrite support
  - Local Chroma vector retrieval
  - Source pack output
  - Source Anchor Pack construction

- 💬 **Upgraded multi-stage dialogue generation pipeline**
  - Initial generation
  - Structural validation
  - Quality critique
  - Content expansion
  - Expression polish
  - Final validation

- ✅ **Added deterministic validation**
  - Round count checks
  - A/B numbering checks
  - Speaker alternation checks
  - Source Appendix checks
  - Short-line ratio and developed-line ratio checks
  - Source / retrieval leakage checks

- 📁 **Improved runtime artifact logging**
  - Prompt JSON
  - Metadata JSON
  - Validation JSON
  - Pipeline run JSON
  - Runtime logs

---

### v0.1.0

- ✨ Initial release
- 🔍 Local RAG retrieval
- 🧠 Local model calls through LM Studio
- 💬 Chinese A/B dialogue generation
- 📦 Basic source pack output
- 🧪 CLI pipeline execution

---

## 🤝 Contributing

Contributions, feedback, and improvement suggestions are welcome.

You can contribute in the following ways:

### Report issues

If you encounter a bug or a generation quality problem, please open a GitHub Issue and include as much relevant information as possible:

- Operating system and Python version
- LM Studio model name
- Run command or UI operation steps
- Error message
- Relevant log files
- Corresponding `pipeline_*.json`
- Generated output or validation report

Issue tracker:

```text
https://github.com/Micheliliuv87/TTSDataGen/issues
```

If your repository name is different, replace the link above with the actual Issues URL.

---

### Improve rule files

Generation behavior is mainly controlled by these rule files:

```text
knowledge_base/rules/generation_rules.jsonl
knowledge_base/rules/critique_rules.jsonl
knowledge_base/rules/expand_rules.jsonl
knowledge_base/rules/polish_rules.jsonl
knowledge_base/rules/validation_rules.jsonl
```

If you notice recurring issues, consider adding or adjusting rules for the corresponding stage.

Suggested test command after making changes:

```bash
bash scripts/run_pipeline.sh "生成30轮A与B对话，主题是城市生活、孤独感和咖啡馆"
```

---

### Improve prompts or configs

Prompt templates and model parameters are mainly under:

```text
configs/
```

Especially:

```text
configs/prompt_templates.yaml
configs/generation.yaml
configs/critic.yaml
configs/expander.yaml
configs/polisher.yaml
configs/rag.yaml
```

After modifying configs, keep the corresponding `.prompt.json` and `.validation.json` files to compare results across versions.

---

### Improve the UI

The Streamlit UI is located at:

```text
app/streamlit_app.py
```

Possible improvements include:

- clearer progress display
- better error messages
- more convenient download buttons
- clearer validation summary
- more stable job cancellation
- better Source Appendix display

---

## 📄 License

Current project license:

[LICENSE](LICENSE) <ins>**All Rights Reserved**</ins>

Unless separately authorized by the author, copying, distribution, modification, republication, or commercial use is not permitted.

For use, citation, derivative development, or commercial collaboration, please contact the author for explicit permission.

---

## 🙏 Acknowledgements

TTSDataGen is supported by the following projects and tools:

- **LM Studio**: local OpenAI-compatible model server
- **Qwen**: local large language model capability
- **BAAI/bge-m3**: local embedding model
- **ChromaDB**: local vector database
- **Streamlit**: local interactive UI
- **HappyScribe public transcript pages**: public transcript source for local RAG data construction
- **Python open-source ecosystem**: data processing, configuration management, and pipeline infrastructure

---

## 📞 Contact

- **Creator**: [Micheli V](https://micheliliuv87.github.io/)
- **GitHub**: [@Micheliliuv87](https://github.com/Micheliliuv87)
- **Issues**: [GitHub Issues](https://github.com/Micheliliuv87/TTSDataGen/issues)

---

<div align="center">

**⭐ If this project is helpful to you, a Star would be appreciated!**

Made with ❤️ by Micheli V

</div>
