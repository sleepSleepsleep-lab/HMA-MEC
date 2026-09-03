#!/bin/bash
set -e

LOG_DIR="../results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log " HMA-MEC 主线实验启动 (E1-E9; E10-E21 见 run_e1x_*.py 单独运行)"
log "=========================================="

log "检查 vLLM 服务..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    log "vLLM 服务正常"
else
    log "警告: vLLM 服务未响应，部分实验可能失败"
fi

run_exp() {
    local name="$1"; shift
    log ""
    log "========== $name =========="
    if python "$@" >> "$LOG" 2>&1; then
        log "$name: 完成"
    else
        log "$name: 失败 (exit=$?)"
    fi
}

cd "$(dirname "$0")"

run_exp "E1 主对比实验"            run_e1_main.py
run_exp "E2 可扩展性实验"          run_e2_scalability.py
run_exp "E3 消融实验"              run_e3_ablation.py
run_exp "E4 推理效率实验"          run_e4_efficiency.py
run_exp "E5 Pareto前沿实验"        run_e5_pareto.py
run_exp "E6 鲁棒性实验"            run_e6_robust.py
run_exp "E7 灵敏度分析"            run_e7_sensitivity.py
run_exp "E8 蒸馏数据量实验"        run_e8_distill_size.py
run_exp "E9 多LLM对比实验"         run_e9_multi_llm.py

log ""
log "=========================================="
log " 全部实验结束，日志: $LOG"
log "=========================================="
