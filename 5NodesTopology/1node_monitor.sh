#!/bin/bash

echo "Starting continuous network RTT monitoring..."
echo "Press Ctrl+C to stop the monitoring"
echo "========================================"

# Trap to handle Ctrl+C gracefully
trap 'echo -e "\n\nMonitoring stopped by user"; exit 0' SIGINT

# Define node mappings
declare -A nodes
nodes["clab-century-serf1"]="10.0.1.11"
nodes["clab-century-serf2"]="10.0.1.12"
nodes["clab-century-serf3"]="10.0.1.13"
nodes["clab-century-serf4"]="10.0.1.14"
nodes["clab-century-serf5"]="10.0.1.15"
nodes["clab-century-serf6"]="10.0.1.16"
nodes["clab-century-serf7"]="10.0.1.17"
nodes["clab-century-serf8"]="10.0.1.18"
nodes["clab-century-serf9"]="10.0.1.19"
nodes["clab-century-serf10"]="10.0.1.20"
nodes["clab-century-serf11"]="10.0.1.21"
nodes["clab-century-serf12"]="10.0.1.22"
nodes["clab-century-serf13"]="10.0.1.23"
nodes["clab-century-serf14"]="10.0.2.24"
nodes["clab-century-serf15"]="10.0.2.25"
nodes["clab-century-serf16"]="10.0.2.26"
nodes["clab-century-serf17"]="10.0.2.27"
nodes["clab-century-serf18"]="10.0.2.28"
nodes["clab-century-serf19"]="10.0.2.29"
nodes["clab-century-serf20"]="10.0.2.30"
nodes["clab-century-serf21"]="10.0.2.31"
nodes["clab-century-serf22"]="10.0.2.32"
nodes["clab-century-serf23"]="10.0.2.33"
nodes["clab-century-serf24"]="10.0.2.34"
nodes["clab-century-serf25"]="10.0.2.35"
nodes["clab-century-serf26"]="10.0.2.36"

# Prompt user for source and destination nodes
read -p "Enter source node (e.g. clab-century-serf1): " src_node
read -p "Enter destination node (e.g. clab-century-serf2): " dst_node

# Validate nodes
if [[ -z "${nodes[$src_node]}" || -z "${nodes[$dst_node]}" ]]; then
    echo "❌ Invalid node name(s). Please check your input."
    exit 1
fi

echo ""
echo "Monitoring RTT from $src_node to $dst_node..."
echo ""

while true; do
    echo "RTT TEST [$src_node ➝ $dst_node]:"
    
    rtt_result=$(./serf_0107 rtt "$src_node" "$dst_node")
    rtt_status=$?
    
    if [ $rtt_status -eq 0 ]; then
        echo "✓ RTT SUCCESS - $rtt_result"
    else
        echo "✗ RTT FAILED - $rtt_result"
    fi

    echo "======================================"
    sleep 1
done
