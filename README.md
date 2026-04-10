<div align="center">

# 🧠 NeuroOS
**The Autonomous Neural Operating System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*NeuroOS is a complete, monolithic AI architecture in a **single Python file** (~1300 lines). It unites local GGUF inference, System 2 reasoning (MCTS), dynamic knowledge injection (LoRA), and continuous learning into one seamless engine.*

[Architecture](#architecture) • [Quick Start](#quick-start) • [Features](#features) • [Why 1 File?](#why-a-single-file)

</div>

---

## 💡 What is NeuroOS?

Most open-source AI projects focus on one piece of the puzzle: fast inference (`vLLM`), agents (`LangChain`), or training (`Unsloth`). **NeuroOS combines them all into a unified "Operating System" for local AI.**

Inspired by OpenAI's o1 and Liquid Neural Networks, NeuroOS doesn't just predict the next token. It **thinks** (Monte Carlo Tree Search), **remembers** (Graph/Vector memory), **debates** (Multi-agent HiveMind), and **evolves** (Continuous LoRA training) — entirely locally on your GPU.

## 🏗️ Architecture (The 5 Pillars)

1. **⚡ Core Engine (NeuroHive GPU):** A native PyTorch GGUF runner. Features PagedAttention, INT8/FP16 KV Cache, RoPE scaling, and GQA support. No C++ compilation required.
2. **🤔 O1-Engine (System 2):** When entropy (uncertainty) is high, the engine pauses token generation and uses **MCTS (Monte Carlo Tree Search)** and Multi-Agent Debate to explore logical pathways before answering.
3. **🧬 ProteusNet & HydraMoE:** The system intercepts user feedback and dynamically hot-swaps LoRA adapters in VRAM. It learns continuously without catastrophic forgetting.
4. **📚 CortexFS:** A hybrid Graph/Vector memory system. The model automatically performs RAG injections into its own context window before generating.
5. **🏭 AutoForge:** An automated pipeline that mines PDFs/Web, generates high-quality synthetic data via self-play, and triggers background LoRA training.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/yourusername/NeuroOS.git
cd NeuroOS
pip install torch numpy gguf tiktoken fastapi uvicorn sentence-transformers networkx
