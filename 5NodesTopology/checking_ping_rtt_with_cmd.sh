#!/bin/bash

echo "Starting continuous network monitoring..."
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

# Counter for iterations
iteration=1

while true; do
    echo ""
    echo "========================================"
    echo "ITERATION #$iteration - $(date)"
    echo "========================================"
    
    # Test each node (ping + RTT side by side)
    for i in {1..26}; do
        node_name="clab-century-serf$i"
        ip_address="${nodes[$node_name]}"
        
        echo "----------------------------------------"
        echo "Node: $node_name ($ip_address)"
        echo "----------------------------------------"
        
        # Ping test
        echo "PING TEST:"
        ping_result=$(ping -c 2 -W 3 $ip_address 2>&1)
        ping_status=$?
        
        if [ $ping_status -eq 0 ]; then
            # Extract average time from ping result (try multiple formats)
            avg_time=$(echo "$ping_result" | grep -E "(round-trip|rtt)" | sed -n 's/.*= [0-9]*\.[0-9]*\/\([0-9]*\.[0-9]*\)\/.*/\1/p')
            if [ -z "$avg_time" ]; then
                # Alternative extraction method
                avg_time=$(echo "$ping_result" | grep -E "time=" | tail -1 | sed 's/.*time=\([0-9]*\.[0-9]*\).*/\1/')
            fi
            if [ -n "$avg_time" ]; then
                echo "✓ PING SUCCESS - Average: ${avg_time}ms"
            else
                echo "✓ PING SUCCESS"
            fi
            # Show the actual ping output for debugging
            echo "$ping_result" | grep -E "(PING|64 bytes|ping statistics|packets transmitted)"
        else
            echo "✗ PING FAILED"
            echo "$ping_result"
        fi
        
        echo ""
        echo "RTT TEST:"
        # Serf RTT test
        rtt_result=$(./serf_2406 rtt clab-century-serf1 $node_name 2>&1)
        rtt_status=$?
        
        if [ $rtt_status -eq 0 ]; then
            echo "✓ RTT SUCCESS - $rtt_result"
        else
            echo "✗ RTT FAILED - $rtt_result"
        fi
        
        echo ""
    done

    echo "======================================"
    echo "Iteration #$iteration completed at $(date)"
    echo "Waiting 5 seconds before next iteration..."
    echo "======================================"
    
    # Wait 30 seconds before next iteration
    sleep 5
    
    # Increment iteration counter
    ((iteration++))
done