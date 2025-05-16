package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
)

type Node struct {
	Name       string
	X          float64
	Y          float64
	RTT        float64
	PingRTT    float64
	PingResult string
}

var nodeIPs = map[string]string{
	"clab-century-serf1":  "10.0.1.11",
	"clab-century-serf2":  "10.0.1.12",
	"clab-century-serf3":  "10.0.1.13",
	"clab-century-serf4":  "10.0.1.14",
	"clab-century-serf5":  "10.0.1.15",
	"clab-century-serf6":  "10.0.1.16",
	"clab-century-serf7":  "10.0.1.17",
	"clab-century-serf8":  "10.0.1.18",
	"clab-century-serf9":  "10.0.1.19",
	"clab-century-serf10": "10.0.1.20",
	"clab-century-serf11": "10.0.1.21",
	"clab-century-serf12": "10.0.1.22",
	"clab-century-serf13": "10.0.1.23",
	"clab-century-serf14": "10.0.1.24",
	"clab-century-serf15": "10.0.1.25",
	"clab-century-serf16": "10.0.1.26",
	"clab-century-serf17": "10.0.1.27",
	"clab-century-serf18": "10.0.1.28",
	"clab-century-serf19": "10.0.1.29",
	"clab-century-serf20": "10.0.1.30",
}

func calculateRTT(a, b *coordinate.Coordinate) float64 {
	if len(a.Vec) != len(b.Vec) {
		panic("coordinate dimensions do not match")
	}
	sumsq := 0.0
	for i := 0; i < len(a.Vec); i++ {
		diff := a.Vec[i] - b.Vec[i]
		sumsq += diff * diff
	}
	rtt := math.Sqrt(sumsq) + a.Height + b.Height
	adjusted := rtt + a.Adjustment + b.Adjustment
	if adjusted > 0.0 {
		rtt = adjusted
	}
	return rtt * 1000 // ms
}

func ping(ip string) (string, float64) {
	cmd := exec.Command("ping", "-c", "1", "-w", "2", ip)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("ping failed: %v", err), -1
	}
	return parsePingOutput(string(output))
}

func parsePingOutput(output string) (string, float64) {
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "time=") {
			timeStr := line[strings.Index(line, "time=")+5:]
			timeStr = strings.TrimSpace(timeStr)
			timeStr = strings.Split(timeStr, " ")[0]
			if ms, err := strconv.ParseFloat(timeStr, 64); err == nil {
				return "time=" + timeStr + " ms", ms
			}
		}
	}
	return "No RTT found", -1
}

func logAndPrint(logger *log.Logger, format string, v ...any) {
	msg := fmt.Sprintf(format, v...)
	fmt.Print(msg)
	logger.Print(msg)
}

func main() {
	logFile, err := os.Create("nodes_log.txt")
	if err != nil {
		log.Fatalf("Could not create log file: %v", err)
	}
	defer logFile.Close()

	logger := log.New(logFile, "", log.LstdFlags)

	serfClient, err := client.NewRPCClient("127.0.0.1:7373")
	if err != nil {
		log.Fatalf("Failed to connect to Serf: %v", err)
	}
	defer serfClient.Close()

	hostname, _ := os.Hostname()

	for {
		members, err := serfClient.Members()
		if err != nil {
			logAndPrint(logger, "Failed to get members: %v\n", err)
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
			nodes = append(nodes, Node{
				Name: member.Name,
				X:    coord.Vec[0],
				Y:    coord.Vec[1],
			})
		}

		if currentNode == "" {
			logAndPrint(logger, "Could not determine the current node\n")
			time.Sleep(2 * time.Second)
			continue
		}

		thisCoord, _ := serfClient.GetCoordinate(currentNode)

		var filtered []Node
		for i := range nodes {
			if nodes[i].Name == currentNode {
				continue
			}
			coord, _ := serfClient.GetCoordinate(nodes[i].Name)
			nodes[i].RTT = calculateRTT(thisCoord, coord)
			ip := nodeIPs[nodes[i].Name]
			nodes[i].PingResult, nodes[i].PingRTT = ping(ip)
			filtered = append(filtered, nodes[i])
		}

		timestamp := time.Now().Format("2006-01-02 15:04:05")
		logAndPrint(logger, "\n--- Run at %s ---\n", timestamp)

		// Coordinates
		logAndPrint(logger, "[COORDINATES]\n")
		logAndPrint(logger, "Node: %-25s => X: %.6f  Y: %.6f  [CURRENT NODE]\n", currentNode, thisCoord.Vec[0], thisCoord.Vec[1])
		for _, n := range filtered {
			logAndPrint(logger, "Node: %-25s => X: %.6f  Y: %.6f\n", n.Name, n.X, n.Y)
		}

		// RTT
		logAndPrint(logger, "\n[RTT]\n")
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].RTT < filtered[j].RTT
		})
		for _, n := range filtered {
			logAndPrint(logger, "Node: %-25s => RTT: %.2f ms\n", n.Name, n.RTT)
		}

		// Ping
		logAndPrint(logger, "\n[PING]\n")
		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].PingRTT < filtered[j].PingRTT
		})
		for _, n := range filtered {
			logAndPrint(logger, "Node: %-25s => Ping: %s\n", n.Name, n.PingResult)
		}

		time.Sleep(2 * time.Second)
	}
}
