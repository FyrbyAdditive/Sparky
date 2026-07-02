# Connecting two DGX Sparks over the ConnectX-7 200GbE link

Follow NVIDIA's `connect-two-sparks` playbook
(https://github.com/NVIDIA/dgx-spark-playbooks) — this is the condensed version
used by the `max-2spark-tp2` profile. Verify each step on your hardware.

## 1. Physical link

Connect a QSFP56 DAC cable directly between a ConnectX-7 port on each Spark
(no switch needed for two nodes).

## 2. Addressing (both Sparks)

Give each CX-7 interface a static IP on a dedicated subnet, e.g.:

```bash
# find the ConnectX interface name
ip link | grep -i -A1 enP                 # often enP2p1s0f0np0 or similar
# Spark A
sudo ip addr add 192.168.100.1/24 dev <cx7-if>
# Spark B
sudo ip addr add 192.168.100.2/24 dev <cx7-if>
sudo ip link set <cx7-if> up mtu 9000
ping 192.168.100.2                        # from A: verify the direct link
```

Make it persistent with netplan/NetworkManager per the playbook.

## 3. RDMA/RoCE sanity checks

```bash
ibstat                                     # link up, rate 200
rdma link show
# bandwidth test (install perftest):
ib_write_bw -d <hca> --report_gbits        # server on B, client on A -> expect ~180+ Gb/s
```

## 4. NCCL over the link

Containers doing multi-node inference need:
- `--device /dev/infiniband` (or compose `devices:` entry)
- `NCCL_SOCKET_IFNAME=<cx7-if>` and typically `NCCL_IB_HCA=<hca>`
- host networking (`network_mode: host`) for rendezvous ports

Validate with the playbook's NCCL test before moving to model serving:
`all_reduce_perf -b 1G -e 4G` across both nodes should saturate the link.

## 5. Serving a TP=2 model

See `deploy/tp2/README.md` for the TRT-LLM (recommended, official playbook)
and vLLM+Ray launch paths.
