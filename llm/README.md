# LLM Setup — RTX 5060 8GB (Blackwell)

## Hardware

- GPU: NVIDIA RTX 5060 8GB (Blackwell, compute capability sm_120)
- VRAM necessária: ~5.4 GB (Qwen2.5-7B Q4_K_M com 32 layers na GPU)
- Compatível também com RTX 4090/4080 (Ada, sm_89)

## Modelo

**Qwen2.5-7B-Instruct Q4_K_M**
- Tamanho: ~5.4 GB
- Contexto: 8192 tokens (suficiente para research_params.py + histórico)
- Qualidade: bom equilíbrio entre velocidade e qualidade de código

## Instalação rápida

```bash
# A partir da raiz do projecto
bash llm/setup.sh
```

Isto irá:
1. Clonar `llama.cpp` de https://github.com/ggerganov/llama.cpp
2. Compilar com CUDA (`-DCMAKE_CUDA_ARCHITECTURES="89;120"`)
3. Fazer download do modelo (~5.4 GB)
4. Testar o servidor

## Iniciar o servidor

Podes usar variáveis de ambiente para tornar os scripts portáteis:

- `CONDA_PREFIX` — prefixo do ambiente Python/Conda a usar (padrão: `$HOME/anaconda3/envs/ml_trading`).
- `MODEL_DIR` — directório onde os modelos `.gguf` estão guardados (padrão: `$HOME/models/gguf`).
- `MODEL_PATH` — caminho absoluto para um modelo específico; tem prioridade sobre o primeiro argumento do script.

```bash
# Exemplo: usar um ambiente e directório de modelos personalizados
CONDA_PREFIX=/opt/miniconda3/envs/ml MODEL_DIR=/data/gguf ./llm/start_server.sh instruct

# Exemplo: forçar um modelo específico via env var
MODEL_PATH=/data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf ./llm/start_server.sh
```

```bash
# Janela 1 — servidor LLM (manter a correr)
./llm/llama.cpp/build/bin/llama-server \
    --model ~/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port 8080 \
    --n-gpu-layers 32 \
    --ctx-size 8192

# Janela 2 — agente
# (active o seu ambiente Python, p.ex.: conda activate ml_trading)
python main.py run
```

## Parâmetros do servidor

| Parâmetro | Valor | Razão |
|-----------|-------|-------|
| `--n-gpu-layers 32` | 32/32 | Modelo cabe inteiro na VRAM |
| `--ctx-size 8192` | 8192 tokens | research_params.py + histórico + prompt |
| `--port 8080` | 8080 | Configurável em config.yaml |

## Verificar se está a correr

```bash
curl http://localhost:8080/health
# Esperar: {"status":"ok"}
```

## Alternativas

Se não tiver GPU com 8GB+, pode usar versão menor:
- **Qwen2.5-3B Q4_K_M** (~2GB VRAM) — qualidade inferior
- **Qwen2.5-7B Q2_K** (~3GB VRAM) — compressão mais agressiva
- **llama.cpp CPU** — remover `--n-gpu-layers 32` (muito mais lento)

## Troubleshooting

```bash
# Verificar CUDA
nvidia-smi

# Verificar compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild se necessário
cd llm/llama.cpp/build && cmake --build . --config Release -j$(nproc)
```
