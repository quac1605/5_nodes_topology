package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/google/hilbert" // Import Hilbert library
	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
)

func main() {
	// Set up logging to a file
	logFile, err := os.OpenFile("hilbert_indices.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}
	defer logFile.Close()
	logger := log.New(logFile, "", log.LstdFlags)

	// Set up the Serf RPC client configuration
	clientConfig := &client.Config{
		Addr: "127.0.0.1:7373",
	}

	serfClient, err := client.ClientFromConfig(clientConfig)
	if err != nil {
		log.Fatalf("Failed to create Serf client: %v", err)
	}
	defer serfClient.Close()

	for {
		clientMembers, err := serfClient.Members()
		if err != nil {
			log.Fatalf("Failed to retrieve members: %v", err)
		}

		for _, member := range clientMembers {
			fmt.Printf("Node: %s, Address: %s:%d, Status: %s, Tags: %v\n",
				member.Name, member.Addr, member.Port, member.Status, member.Tags)

			// Fetch the network coordinate for the member
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil {
				fmt.Printf("Failed to get coordinate for node %s: %v\n", member.Name, err)
				continue
			}

			// Retrieve CPU and memory resources from tags (example assumption)
			cpu := parseFloat(member.Tags["cpu"], 0)
			memory := parseFloat(member.Tags["memory"], 0)

			// Normalize values and calculate Hilbert index
			hilbertIndex := calculateHilbertIndex(coord, cpu, memory)
			fmt.Printf("Hilbert Index for %s: %d\n", member.Name, hilbertIndex)

			// Log Hilbert index
			logger.Printf("Time: %s - Hilbert Index for %s: %d\n",
				time.Now().Format(time.RFC3339), member.Name, hilbertIndex)
		}

		time.Sleep(10 * time.Second)
	}
}

// calculateHilbertIndex calculates the 4D Hilbert index for a node
func calculateHilbertIndex(coord *coordinate.Coordinate, cpu, memory float64) uint64 {
	// Define scaling factors (these can be adjusted based on the dataset)
	scaleFactor := 1000.0
	cpuMax := 4.0       // Assume max CPU is 4 cores
	memoryMax := 8192.0 // Assume max memory is 8GB

	// Normalize values
	x := int(math.Round(coord.Vec[0] * scaleFactor))
	y := int(math.Round(coord.Vec[1] * scaleFactor))
	z := int(math.Round((cpu / cpuMax) * scaleFactor))
	w := int(math.Round((memory / memoryMax) * scaleFactor))

	// Calculate Hilbert index
	hilbertOrder := 10 // Number of bits per dimension
	index, _ := hilbert.Encode(hilbertOrder, []int{x, y, z, w})

	return index
}

// parseFloat safely parses a string to float, returns default value on failure
func parseFloat(s string, defaultValue float64) float64 {
	if val, err := strconv.ParseFloat(s, 64); err == nil {
		return val
	}
	return defaultValue
}
