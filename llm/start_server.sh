#!/usr/bin/env bash
# start_server.sh — Inicia o llama-server com as libs CUDA do conda
#
# Uso:
#   ./llm/start_server.sh                    # modelo padrão (instruct)
#   ./llm/start_server.sh coder              # Qwen2.5-Coder-7B Q4_K_M
#   ./llm/start_server.sh instruct           # Qwen2.5-7B-Instruct Q4_K_M
#   ./llm/start_server.sh /path/to/model.gguf  # path absoluto
#   ./llm/start_server.sh coder 8081         # modelo coder no porto 8081

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_PREFIX:-$HOME/anaconda3/envs/ml_trading}"
SERVER="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"
MODELS_DIR="${MODEL_DIR:-$HOME/models/gguf}"
PORT="${2:-8080}"
MODEL_PATH="${MODEL_PATH:-}"

# --- Modelos disponíveis ---
MODELS=(
    "instruct:$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf:Qwen2.5-7B-Instruct Q4_K_M (geral, 4.4GB)"
    "coder:$MODELS_DIR/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf:Qwen2.5-Coder-7B-Instruct Q4_K_M (código, 4.4GB)"
)

# --- Selecção do modelo ---
MODEL_ARG="${1:-instruct}"

if [[ -n "$MODEL_PATH" ]]; then
    # MODEL_PATH tem prioridade sobre argumentos
    MODEL="$MODEL_PATH"
    MODEL_DESC="custom (MODEL_PATH): $MODEL"
elif [[ "$MODEL_ARG" == /* || "$MODEL_ARG" == ~* || "$MODEL_ARG" == ./* ]]; then
    # Path absoluto/relativo passado directamente
    MODEL="$MODEL_ARG"
    MODEL_DESC="custom: $MODEL"
else
    # Procurar por nome curto na lista
    MODEL=""
    MODEL_DESC=""
    for entry in "${MODELS[@]}"; do
        name="${entry%%:*}"
        rest="${entry#*:}"
        path="${rest%%:*}"
        desc="${rest#*:}"
        if [[ "$name" == "$MODEL_ARG" ]]; then
            MODEL="$path"
            MODEL_DESC="$desc"
            break
        fi
    done

    if [[ -z "$MODEL" ]]; then
        echo "ERRO: modelo '$MODEL_ARG' não reconhecido."
        echo ""
        echo "Modelos disponíveis:"
        for entry in "${MODELS[@]}"; do
            name="${entry%%:*}"
            rest="${entry#*:}"
            path="${rest%%:*}"
            desc="${rest#*:}"
            exists="✓"
            [[ ! -f "$path" ]] && exists="✗ NÃO DESCARREGADO"
            printf "  %-12s %s  [%s]\n" "$name" "$desc" "$exists"
        done
        echo ""
        echo "Para descarregar modelos em falta: bash llm/setup.sh --download-all"
        exit 1
    fi
fi

# --- Verificar se o ficheiro existe ---
if [[ ! -f "$MODEL" ]]; then
    echo "ERRO: modelo não encontrado: $MODEL"
    echo "Descarregar com: bash llm/setup.sh --download-all"
    exit 1
fi

export LD_LIBRARY_PATH="$SCRIPT_DIR/llama.cpp/build/bin:$CONDA_ENV/lib:$CONDA_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "═══════════════════════════════════════════════"
echo "  Servidor LLM"
echo "  Modelo: $MODEL_DESC"
echo "  Porto:  $PORT"
echo "  VRAM:   $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) MB livres"
echo "═══════════════════════════════════════════════"
echo ""

"$SERVER" \
    --model "$MODEL" \
    --port "$PORT" \
    --n-gpu-layers 32 \
    --ctx-size 16384 \
    --flash-attn on \
    --parallel 1
