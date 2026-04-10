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
```
### 2. Download a GGUF Model

NeuroOS works beautifully with Llama-3 (Q4_K_M is recommended for 8GB+ VRAM).

```
wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```
### 3. Run the OS

Interactive Chat Mode (System 1):

```
python neuroos.py --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --interactive
```

Deep Reasoning Mode (System 2 / o1-style):

```
python neuroos.py --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --prompt "Solve P=NP" --deep-think
```

Start the OpenAI-Compatible API Server:

```
python neuroos.py --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --api --api-port 8000
```

You can now connect LangChain, AutoGen, or UI frontends (like SillyTavern) to http://localhost:8000/v1.

## 🛠️ CLI Showcase

NeuroOS is packed with features accessible directly from the terminal:

Multi-Agent Debate: Force the model to clone itself into personas (Logic, Creative, Skeptic) to debate a topic before giving you the final answer.

```
python neuroos.py --model model.gguf --prompt "Should we colonize Mars?" --debate
```

Mine Knowledge into CortexFS:

```
python neuroos.py --model model.gguf --mine-pdf /path/to/quantum_physics.pdf
```

AutoForge (Generate Data & Queue LoRA Training):

```
python neuroos.py --model model.gguf --forge --topics "Rust Async Programming, Fastapi"
```

## 📜 Why a single file?

In the spirit of Andrej Karpathy's micrograd and llm.c, NeuroOS is contained entirely within neuroos.py.

Modern AI stacks have become a labyrinth of fragmented libraries, Docker containers, and complex C++ dependencies. NeuroOS serves as an educational masterpiece and architectural blueprint. By putting the entire AGI stack (Inference, RAG, MCTS, LoRA routing) in one readable Python file, researchers and hackers can clearly see how the boundaries between "Memory", "Reasoning", and "Generation" interact at the tensor level.

It is a hacker's playground. Fork it, break it, learn from it.

## 🤝 Contributing

NeuroOS is an experimental Proof of Concept. PRs are welcome! Areas of active development:

Native Triton kernels for faster Q4_0 dequantization on GPU.

Integration with unsloth for the AutoForge background training loop.

Expanding the Symbolic Bridge (Fuzzy Logic constraints).

## 📄 License

MIT License.
