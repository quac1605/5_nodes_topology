#!/bin/sh

# List of nodes
NODES="clab-century-serf1 clab-century-serf2 clab-century-serf3 clab-century-serf4 clab-century-serf5 clab-century-serf6 clab-century-serf7 clab-century-serf8 clab-century-serf9 clab-century-serf10 clab-century-serf11 clab-century-serf12 clab-century-serf13 clab-century-serf14 clab-century-serf15 clab-century-serf16 clab-century-serf177 clab-century-serf18 clab-century-serf19 clab-century-serf20"

# Loop through all nodes and run the command
for NODE in $NODES; do
    echo "Checking Serf members on $NODE..."
    docker exec -it $NODE sh -c 'export PATH=$PATH:/opt/serfapp && serf members -detailed'
    echo "--------------------------------------"
done
