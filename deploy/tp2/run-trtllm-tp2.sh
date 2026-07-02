#!/usr/bin/env bash
# Launch a TP=2 TRT-LLM OpenAI server across two Sparks over the RoCE link.
# Run from Spark A (rank 0) after completing deploy/interconnect.md.
# Verify against NVIDIA/dgx-spark-playbooks (connect-two-sparks + trt-llm
# recipes) for the container tag and flag set matching your installation.
set -euo pipefail

MODEL="${TP2_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP4}"
IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
SPARK_A_IB="${SPARK_A_IB:-192.168.100.1}"
SPARK_B_IB="${SPARK_B_IB:-192.168.100.2}"
CX7_IF="${CX7_IF:-enP2p1s0f0np0}"

HOSTFILE=$(mktemp)
printf '%s slots=1\n%s slots=1\n' "$SPARK_A_IB" "$SPARK_B_IB" > "$HOSTFILE"
echo "Hostfile:" && cat "$HOSTFILE"

# Both nodes must have $IMAGE pulled and this repo at the same path.
mpirun -np 2 --hostfile "$HOSTFILE" \
  -x NCCL_SOCKET_IFNAME="$CX7_IF" \
  -x NCCL_IB_HCA \
  docker run --rm --gpus all --network host --ipc host \
    --device /dev/infiniband \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN \
    "$IMAGE" \
    trtllm-serve "$MODEL" \
      --host 0.0.0.0 --port 8010 \
      --tp_size 2 \
      --max_batch_size "${TP2_MAX_BATCH:-8}" \
      --trust_remote_code
