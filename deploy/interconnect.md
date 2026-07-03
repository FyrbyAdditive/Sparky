# Connecting the two Sparks over the ConnectX-7 200GbE link

The bot's inference traffic (magi → shodan: LLM + wiki calls) runs over a
direct ConnectX-7 link on 192.168.100.0/24 — both machines' default routes
are WiFi, so this link is what makes the split fast. Based on NVIDIA's
`connect-two-sparks` playbook (https://github.com/NVIDIA/dgx-spark-playbooks).

## 1. Physical link

Connect a QSFP56 DAC cable directly between a ConnectX-7 port on each Spark
(no switch needed for two nodes).

## 2. Addressing (both Sparks)

Give each CX-7 interface a static IP on the dedicated subnet:

```bash
# find the ConnectX interface name — it varies by slot/firmware:
# observed enp1s0f1np1 on this pair, enP2p1s0f0np0 on others
ip link | grep -i -E "enp|enP"
# magi
sudo ip addr add 192.168.100.1/24 dev <cx7-if>
# shodan
sudo ip addr add 192.168.100.2/24 dev <cx7-if>
sudo ip link set <cx7-if> up mtu 9000
ping 192.168.100.2                        # from magi: verify the direct link
```

Persist with netplan — `interconnect-netplan.yaml` in this directory is the
config in use (edit the interface name to match yours). Measured on this
pair: 70.9 Gbit/s TCP, ample for the ~KB-scale request traffic.

The bot host env (`profiles/magi.bot.env`) points at shodan via
`SPARK_B_IB=192.168.100.2`.
