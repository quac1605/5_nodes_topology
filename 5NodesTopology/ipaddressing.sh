#!/bin/sh

# Configure serf1–serf10: subnet 10.0.1.0/24
for i in $(seq 1 10)
do
  sudo docker exec -d clab-century-serf$i ip link set eth1 up
  sudo docker exec -d clab-century-serf$i ip addr add 10.0.1.$((10+i))/24 brd 10.0.1.255 dev eth1
  sudo docker exec -d clab-century-serf$i ip route del default via 172.20.20.1 dev eth0
  sudo docker exec -d clab-century-serf$i ip route add default via 10.0.1.1 dev eth1
done

# Configure serf11–serf20: subnet 10.0.2.0/24
for i in $(seq 11 20)
do
  ip_suffix=$((10 + i))  # So serf11 gets 10.0.2.21, serf12 gets 10.0.2.22, etc.
  sudo docker exec -d clab-century-serf$i ip link set eth1 up
  sudo docker exec -d clab-century-serf$i ip addr add 10.0.2.${ip_suffix}/24 brd 10.0.2.255 dev eth1
  sudo docker exec -d clab-century-serf$i ip route del default via 172.20.20.1 dev eth0
  sudo docker exec -d clab-century-serf$i ip route add default via 10.0.2.1 dev eth1
done
