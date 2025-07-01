package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/hashicorp/serf/client"
)

type Node struct {
	Name   string
	X      float64
	Y      float64
	Z      float64
	RTT    float64
	RAM    string
	Vcores int
}

var nodeResources = map[string]struct {
	RAM    string
	Vcores int
}{
	"clab-century-serf1":  {RAM: "16GB", Vcores: 4},
	"clab-century-serf2":  {RAM: "32GB", Vcores: 8},
	"clab-century-serf3":  {RAM: "8GB", Vcores: 2},
	"clab-century-serf4":  {RAM: "64GB", Vcores: 12},
	"clab-century-serf5":  {RAM: "16GB", Vcores: 6},
	"clab-century-serf6":  {RAM: "32GB", Vcores: 10},
	"clab-century-serf7":  {RAM: "8GB", Vcores: 2},
	"clab-century-serf8":  {RAM: "64GB", Vcores: 16},
	"clab-century-serf9":  {RAM: "16GB", Vcores: 4},
	"clab-century-serf10": {RAM: "32GB", Vcores: 6},
	"clab-century-serf11": {RAM: "8GB", Vcores: 2},
	"clab-century-serf12": {RAM: "64GB", Vcores: 14},
	"clab-century-serf13": {RAM: "16GB", Vcores: 4},
	"clab-century-serf14": {RAM: "32GB", Vcores: 8},
	"clab-century-serf15": {RAM: "8GB", Vcores: 2},
	"clab-century-serf16": {RAM: "64GB", Vcores: 12},
	"clab-century-serf17": {RAM: "16GB", Vcores: 6},
	"clab-century-serf18": {RAM: "32GB", Vcores: 8},
	"clab-century-serf19": {RAM: "8GB", Vcores: 2},
	"clab-century-serf20": {RAM: "64GB", Vcores: 16},
	"clab-century-serf21": {RAM: "16GB", Vcores: 6},
	"clab-century-serf22": {RAM: "32GB", Vcores: 10},
	"clab-century-serf23": {RAM: "8GB", Vcores: 2},
	"clab-century-serf24": {RAM: "64GB", Vcores: 14},
	"clab-century-serf25": {RAM: "16GB", Vcores: 4},
	"clab-century-serf26": {RAM: "32GB", Vcores: 6},
}

func getRTTFromCommand(source, target string) (float64, error) {
	cmd := exec.Command("./serf_0107", "rtt", source, target)
	output, err := cmd.Output()
	if err != nil {
		return -1, fmt.Errorf("failed to run serf_og rtt command: %w", err)
	}

	line := strings.TrimSpace(string(output))
	parts := strings.Split(line, "rtt:")
	if len(parts) < 2 {
		return -1, fmt.Errorf("unexpected output format: %s", line)
	}

	rttStr := strings.TrimSpace(parts[1])
	rttStr = strings.TrimSuffix(rttStr, " ms")

	rttVal, err := strconv.ParseFloat(rttStr, 64)
	if err != nil {
		return -1, fmt.Errorf("failed to parse RTT value: %w", err)
	}

	return rttVal, nil
}

func logAndPrint(logger *log.Logger, format string, v ...any) {
	msg := fmt.Sprintf(format, v...)
	fmt.Print(msg)
	logger.Print(msg)
}

func main() {
	serfClient, err := client.NewRPCClient("127.0.0.1:7373")
	if err != nil {
		log.Fatalf("Failed to connect to Serf: %v", err)
	}
	defer serfClient.Close()

	hostname, _ := os.Hostname()

	for {
		logFile, err := os.Create("nodes_log.txt")
		if err != nil {
			log.Fatalf("Could not create log file: %v", err)
		}
		logger := log.New(logFile, "", log.LstdFlags)

		members, err := serfClient.Members()
		if err != nil {
			logAndPrint(logger, "Failed to get members: %v\n", err)
			logFile.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		var currentNode string
		var nodes []Node

		for _, member := range members {
			if strings.HasSuffix(member.Name, hostname) {
				currentNode = member.Name
			}
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil || coord == nil {
				continue
			}

			res := nodeResources[member.Name]

			nodes = append(nodes, Node{
				Name:   member.Name,
				X:      coord.Vec[0],
				Y:      coord.Vec[1],
				Z:      coord.Vec[2],
				RAM:    res.RAM,
				Vcores: res.Vcores,
			})
		}

		if currentNode == "" {
			logAndPrint(logger, "Could not determine the current node\n")
			logFile.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		thisCoord, _ := serfClient.GetCoordinate(currentNode)

		var filtered []Node
		for i := range nodes {
			if nodes[i].Name == currentNode {
				continue
			}

			rtt, err := getRTTFromCommand(currentNode, nodes[i].Name)
			if err != nil {
				logAndPrint(logger, "RTT error for %s: %v\n", nodes[i].Name, err)
				rtt = -1
			}
			nodes[i].RTT = rtt

			filtered = append(filtered, nodes[i])
		}

		timestamp := time.Now().Format("2006-01-02 15:04:05")
		logAndPrint(logger, "\n--- Run at %s ---\n", timestamp)

		// Coordinates + Resources
		logAndPrint(logger, "[COORDINATES + RESOURCES]\n")
		logAndPrint(logger, "Node: %-25s => X: %.6f  Y: %.6f  Z: %.6f  RAM: %-5s  vCores: %d   [CURRENT NODE]\n",
			currentNode, thisCoord.Vec[0], thisCoord.Vec[1], thisCoord.Vec[2],
			nodeResources[currentNode].RAM, nodeResources[currentNode].Vcores)

		for _, n := range filtered {
			logAndPrint(logger, "Node: %-25s => X: %.6f  Y: %.6f  Z: %.6f  RAM: %-5s  vCores: %d\n",
				n.Name, n.X, n.Y, n.Z, n.RAM, n.Vcores)
		}

		// RTT
		logAndPrint(logger, "\n[RTT]\n")
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].RTT < filtered[j].RTT
		})
		for _, n := range filtered {
			logAndPrint(logger, "Node: %-25s => RTT: %.2f ms\n", n.Name, n.RTT)
		}

		logFile.Close()
		time.Sleep(8 * time.Second)
	}
}
