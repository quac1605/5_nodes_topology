#!/bin/bash

# Function to apply tc delay safely
set_tc_delay() {
  local iface=$1
  local delay=$2
  echo "Setting tc delay on $iface to $delay"
  sudo tc qdisc del dev "$iface" root 2>/dev/null
  sudo tc qdisc add dev "$iface" root netem delay "$delay"
}

# OVS ports: set delay
set_tc_delay ovs1p1 25ms
set_tc_delay ovs2p1 25ms

for i in {2..11}; do
  set_tc_delay ovs1p$i ${i}ms
  set_tc_delay ovs2p$i ${i}ms
done

# Containerlab netem settings
echo "Applying containerlab netem delays..."
sudo containerlab tools netem set --node clab-century-router1 --interface eth1 --delay 25ms
sudo containerlab tools netem set --node clab-century-router1 --interface eth2 --delay 25ms

for i in {1..10}; do
  sudo containerlab tools netem set --node clab-century-serf$i --interface eth1 --delay $((i+1))ms
done

for i in {11..20}; do
  sudo containerlab tools netem set --node clab-century-serf$i --interface eth1 --delay $((i-9))ms
done
