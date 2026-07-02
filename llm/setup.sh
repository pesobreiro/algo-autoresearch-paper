#!/usr/bin/env bash
# setup.sh — Build llama.cpp (CUDA sm_89 + sm_120) e download Qwen2.5-7B Q4_K_M
# Hardware: RTX 5060 8GB (Blackwell, sm_120)
# Compatível também com RTX 4090/4080 (Ada, sm_89)
#
# Uso: bash llm/setup.sh
# Tempo estimado: 10-20 min (build) + 5-10 min (download ~5.4GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
MODELS_DIR="${MODEL_DIR:-$HOME/models/gguf}"

echo "═══════════════════════════════════════════════"
echo "algo_autoresearch — Setup llama.cpp + Qwen2.5"
echo "═══════════════════════════════════════════════"
echo "Project: $PROJECT_DIR"
echo "llama.cpp: $LLAMA_DIR"
echo "Models: $MODELS_DIR"
echo ""

# --- 1. Clonar llama.cpp ---
if [ ! -d "$LLAMA_DIR" ]; then
    echo "[1/4] A clonar llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
    echo "[1/4] llama.cpp já existe, a atualizar..."
    cd "$LLAMA_DIR" && git pull
fi

# --- 2. Build com CUDA ---
echo ""
echo "[2/4] A compilar com CUDA (sm_89 para RTX 4090; sm_120 para RTX 5060)..."
echo "      Isto pode demorar 10-20 minutos..."

# Usar CUDA toolkit do ambiente conda ml_trading
CONDA_ENV="${CONDA_PREFIX:-$HOME/anaconda3/envs/ml_trading}"
if [ -f "$CONDA_ENV/bin/nvcc" ]; then
    export CUDA_TOOLKIT_ROOT_DIR="$CONDA_ENV"
    export PATH="$CONDA_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    echo "  Usando nvcc: $CONDA_ENV/bin/nvcc"
    "$CONDA_ENV/bin/nvcc" --version | head -1
else
    echo "ERRO: nvcc não encontrado em $CONDA_ENV/bin/"
    echo "Instalar com: conda install -n ml_trading -c nvidia cuda-toolkit"
    exit 1
fi

# Limpar build anterior (evita cache corrompido)
rm -rf "$LLAMA_DIR/build"
mkdir -p "$LLAMA_DIR/build"
cd "$LLAMA_DIR/build"

cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89;120" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER="$CONDA_ENV/bin/nvcc" \
    2>&1 | tail -30

cmake --build . --config Release -j$(nproc)

if [ ! -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo "ERRO: llama-server não foi compilado!"
    echo "Verificar instalação CUDA e tentar novamente."
    exit 1
fi

echo "✓ llama-server compilado: $LLAMA_DIR/build/bin/llama-server"

# --- 3. Criar directório de modelos ---
mkdir -p "$MODELS_DIR"

# --- Função de download ---
download_model() {
    local file="$1"
    local url="$2"
    local desc="$3"

    if [ -f "$file" ]; then
        echo "  ✓ Já existe: $(basename "$file") ($(du -sh "$file" | cut -f1))"
    else
        echo "  A descarregar: $desc"
        echo "  URL: $url"
        if command -v wget &>/dev/null; then
            wget -c -O "$file" --progress=bar:force "$url"
        elif command -v curl &>/dev/null; then
            curl -L -C - -o "$file" --progress-bar "$url"
        else
            echo "ERRO: wget ou curl necessário"
            exit 1
        fi
        echo "  ✓ Descarregado: $(basename "$file") ($(du -sh "$file" | cut -f1))"
    fi
}

# --- 4. Download dos modelos ---
echo ""
echo "[3/4] A verificar/descarregar modelos..."
echo ""

# Qwen2.5-7B-Instruct Q4_K_M — geral (padrão)
download_model \
    "$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
    "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
    "Qwen2.5-7B-Instruct Q4_K_M (~4.4GB, geral)"

# Qwen2.5-Coder-7B-Instruct Q4_K_M — especializado em código
# Descarregar apenas se --download-all passado, ou se o ficheiro não existir e for pedido
if [[ "${1:-}" == "--download-all" ]] || [[ "${DOWNLOAD_CODER:-0}" == "1" ]]; then
    download_model \
        "$MODELS_DIR/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" \
        "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" \
        "Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.4GB, código)"
else
    coder_file="$MODELS_DIR/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    if [ -f "$coder_file" ]; then
        echo "  ✓ Já existe: $(basename "$coder_file") ($(du -sh "$coder_file" | cut -f1))"
    else
        echo "  ○ Qwen2.5-Coder-7B-Instruct Q4_K_M — não descarregado"
        echo "    Para descarregar: bash llm/setup.sh --download-all"
    fi
fi

# --- 5. Teste rápido ---
echo ""
echo "[4/4] A testar llama-server..."
"$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -3

echo ""
echo "═══════════════════════════════════════════════"
echo "✓ Setup concluído!"
echo ""
echo "Modelos disponíveis em $MODELS_DIR:"
for f in "$MODELS_DIR"/*.gguf; do
    [ -f "$f" ] && echo "  $(du -sh "$f" | cut -f1)  $(basename "$f")"
done
echo ""
echo "Para iniciar o servidor LLM (janela separada):"
echo ""
echo "  # Modelo geral (padrão):"
echo "  bash llm/start_server.sh instruct"
echo ""
echo "  # Modelo especializado em código:"
echo "  bash llm/start_server.sh coder"
echo ""
echo "Depois, numa nova janela:"
echo "  # Active o seu ambiente Python (p.ex.: conda activate ml_trading)"
echo "  python main.py run"
echo "═══════════════════════════════════════════════"
