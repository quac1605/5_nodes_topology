package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
)

// IP mapping for nodes
var nodeIPs = map[string]string{
	"clab-century-serf1": "10.0.1.11",
	"clab-century-serf2": "10.0.1.12",
	"clab-century-serf3": "10.0.1.13",
	"clab-century-serf4": "10.0.1.14",
	"clab-century-serf5": "10.0.1.15",
}

func main() {
	// Log setup
	logFile, err := os.OpenFile("rtt_estimates.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()
	logger := log.New(logFile, "", log.LstdFlags)

	// Connect to Serf
	clientConfig := &client.Config{
		Addr: "127.0.0.1:7373",
	}
	serfClient, err := client.ClientFromConfig(clientConfig)
	if err != nil {
		log.Fatalf("Failed to create Serf client: %v", err)
	}
	defer serfClient.Close()

	// Get local hostname
	hostname, err := os.Hostname()
	if err != nil {
		log.Fatalf("Failed to get hostname: %v", err)
	}

	// Find this node's full Serf name
	clientMembers, err := serfClient.Members()
	if err != nil {
		log.Fatalf("Failed to retrieve Serf members: %v", err)
	}

	var thisNode string
	for _, member := range clientMembers {
		if strings.HasSuffix(member.Name, hostname) {
			thisNode = member.Name
			break
		}
	}
	if thisNode == "" {
		log.Fatalf("Failed to match hostname '%s' with any Serf node name", hostname)
	}

	fmt.Printf("This node is: %s (hostname: %s)\n", thisNode, hostname)

	// Main loop
	for {
		clientMembers, err := serfClient.Members()
		if err != nil {
			log.Printf("Failed to retrieve members: %v", err)
			continue
		}

		thisCoord, err := serfClient.GetCoordinate(thisNode)
		if err != nil {
			log.Printf("Failed to get local coordinate: %v", err)
			continue
		}

		for _, member := range clientMembers {
			if member.Name == thisNode {
				continue
			}

			otherCoord, err := serfClient.GetCoordinate(member.Name)
			if err != nil {
				fmt.Printf("Failed to get coordinate for %s: %v\n", member.Name, err)
				continue
			}

			// Estimated RTT
			serfRTT := calculateRTT(thisCoord, otherCoord)
			fmt.Printf("Estimated RTT from %s to %s: %.2f ms\n", thisNode, member.Name, serfRTT)
			logger.Printf("Estimated RTT from %s to %s: %.2f ms\n", thisNode, member.Name, serfRTT)

			// Ping RTT
			ip, ok := nodeIPs[member.Name]
			if !ok {
				fmt.Printf("No IP mapping for %s\n", member.Name)
				continue
			}
			pingRTT := ping(ip)
			fmt.Printf("Ping RTT from %s to %s (%s): %s\n", thisNode, member.Name, ip, pingRTT)
			logger.Printf("Ping RTT from %s to %s (%s): %s\n", thisNode, member.Name, ip, pingRTT)
		}

		time.Sleep(10 * time.Second)
	}
}

// Serf RTT using Vivaldi coordinates
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
	return rtt * 1000 // milliseconds
}

// Run ping and return RTT result
func ping(ip string) string {
	cmd := exec.Command("ping", "-c", "1", "-w", "2", ip)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("ping failed: %v", err)
	}
	return parsePingOutput(string(output))
}

// Parse ping output to extract RTT
func parsePingOutput(output string) string {
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "time=") {
			return strings.TrimSpace(line[strings.Index(line, "time="):])
		}
	}
	return "No RTT found"
}
