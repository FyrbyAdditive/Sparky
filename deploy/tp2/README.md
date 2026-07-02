# Profile max-2spark-tp2 — one big model across both Sparks

Shards a 235B-class MoE across both Sparks (tensor parallel, TP=2) over the
ConnectX-7 RoCE link. Complete `deploy/interconnect.md` first and verify the
NCCL all-reduce test before serving.

Default model: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP4` — demonstrated by
NVIDIA's dual-Spark TRT-LLM playbook. Command details below follow the
`dgx-spark-playbooks` recipes; **verify flags against the playbook version
matching your installed containers** — these launch paths move fast.

## Path A (recommended): TRT-LLM + MPI

On **both** Sparks, pull the TRT-LLM release container used by the playbook
(e.g. `nvcr.io/nvidia/tensorrt-llm/release:<tag>`; the photo-booth used a
`spark-single-gpu-dev` tag — for multi-node use the playbook's tag).

1. Set up passwordless SSH between the Sparks over the 192.168.100.x link.
2. On Spark A (rank 0), run `./run-trtllm-tp2.sh` (this directory). It:
   - creates a hostfile with both CX-7 IPs
   - launches the container on both nodes via `mpirun -np 2 --hostfile ...`
   - starts `trtllm-serve <model> --tp_size 2 --host 0.0.0.0` (OpenAI API on :8000)
3. Point the bot profile at Spark A: `deploy/profiles/max-2spark-tp2.bot.env`.

Optional: the playbook also demonstrates Eagle3 speculative decoding across
two nodes — worth benchmarking for voice-turn latency.

## Path B (alternative): vLLM + Ray

1. Start a Ray head on Spark A and worker on Spark B (inside the vLLM
   container, `network_mode: host`, `/dev/infiniband` mapped,
   `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` set — see interconnect.md).
2. `vllm serve <model> --tensor-parallel-size 2 --distributed-executor-backend ray`

Community references for exact working setups on Spark:
- github.com/eugr/spark-vllm-docker
- github.com/mark-ramsey-ri/vllm-dgx-spark

## Speech/vision/router placement

TP=2 consumes both GPUs' memory bandwidth during decode. Run the aux services
(Riva, Kokoro, vision, router, wiki) on whichever Spark has headroom — start
them with the spark-b compose (`deploy/spark-b/`) and expect some contention;
benchmark with `scripts/bench.py` before judging the profile.
