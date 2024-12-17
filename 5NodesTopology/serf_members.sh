#!/bin/sh

# List of nodes
NODES="clab-century-serf1 clab-century-serf2 clab-century-serf3 clab-century-serf4 clab-century-serf5"

# Loop through all nodes and run the command
for NODE in $NODES; do
    echo "Checking Serf members on $NODE..."
    docker exec -it $NODE sh -c 'export PATH=$PATH:/opt/serfapp && serf members'
    echo "--------------------------------------"
done
