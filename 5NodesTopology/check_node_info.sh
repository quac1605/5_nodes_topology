#!/bin/sh

# List of nodes
NODES="clab-century-serf1 clab-century-serf2 clab-century-serf3 clab-century-serf4 clab-century-serf5"

# Loop through all nodes and check Vivaldi coordinates
for NODE in $NODES; do
    echo "Checking Vivaldi coordinates on $NODE..."
    docker exec -it $NODE sh -c 'export PATH=$PATH:/opt/serfapp && serf info'
    echo "--------------------------------------"
done
