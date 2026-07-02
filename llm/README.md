# LLM Setup — RTX 5060 8GB (Blackwell)

## Hardware

- GPU: NVIDIA RTX 5060 8GB (Blackwell, compute capability sm_120)
- Required VRAM: ~5.4 GB (Qwen2.5-7B Q4_K_M with 32 layers on GPU)
- Also compatible with RTX 4090/4080 (Ada, sm_89)

## Model

**Qwen2.5-7B-Instruct Q4_K_M**
- Size: ~5.4 GB
- Context: 8192 tokens (enough for research_params.py + history)
- Quality: good balance between speed and code quality

## Quick install

```bash
# From the project root
bash llm/setup.sh
```

This will:
1. Clone `llama.cpp` from https://github.com/ggerganov/llama.cpp
2. Compile with CUDA (`-DCMAKE_CUDA_ARCHITECTURES="89;120"`)
3. Download the model (~5.4 GB)
4. Test the server

## Starting the server

You can use environment variables to make the scripts portable:

- `CONDA_PREFIX` — Python/Conda environment prefix to use (default: `$HOME/anaconda3/envs/ml_trading`).
- `MODEL_DIR` — directory where `.gguf` models are stored (default: `$HOME/models/gguf`).
- `MODEL_PATH` — absolute path to a specific model; takes priority over the script's first argument.

```bash
# Example: use a custom environment and model directory
CONDA_PREFIX=/opt/miniconda3/envs/ml MODEL_DIR=/data/gguf ./llm/start_server.sh instruct

# Example: force a specific model via env var
MODEL_PATH=/data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf ./llm/start_server.sh
```

```bash
# Window 1 — LLM server (keep running)
./llm/llama.cpp/build/bin/llama-server \
    --model ~/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    --n-gpu-layers 32 \
    --ctx-size 8192

# Window 2 — agent
# (activate your Python environment, e.g. conda activate ml_trading)
python main.py run
```

## Server parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--n-gpu-layers 32` | 32/32 | Model fits entirely in VRAM |
| `--ctx-size 8192` | 8192 tokens | research_params.py + history + prompt |
| `--port 8080` | 8080 | Configurable in config.yaml |

## Check it is running

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

## Alternatives

If you do not have a GPU with 8GB+, you can use a smaller version:
- **Qwen2.5-3B Q4_K_M** (~2GB VRAM) — lower quality
- **Qwen2.5-7B Q2_K** (~3GB VRAM) — more aggressive compression
- **llama.cpp CPU** — remove `--n-gpu-layers 32` (much slower)

## Troubleshooting

```bash
# Check CUDA
nvidia-smi

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild if needed
cd llm/llama.cpp/build && cmake --build . --config Release -j$(nproc)
```
