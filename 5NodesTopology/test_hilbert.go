package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hashicorp/serf/client"
)

type NodeResources struct {
	RemainingCPU    float64
	RemainingMemory float64 // Store memory in MiB for uniformity
}

func main() {
	// Set up logging to a file
	logFile, err := os.OpenFile("node_data.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
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
		// Load node resources from the file
		nodeResources, err := loadNodeResources("container_stats.txt")
		if err != nil {
			log.Fatalf("Failed to load node resources: %v", err)
		}
		clientMembers, err := serfClient.Members()
		if err != nil {
			log.Fatalf("Failed to retrieve members: %v", err)
		}

		for _, member := range clientMembers {
			// Print basic node information
			fmt.Printf("Node Name: %s\n\tAddress: %s:%d\n\tStatus: %s\n\tTags: %v\n",
				member.Name, member.Addr, member.Port, member.Status, member.Tags)

			// Fetch the network coordinate for the member
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil {
				fmt.Printf("\tFailed to get coordinate for node %s: %v\n", member.Name, err)
				continue
			}

			// Get the node's resource data
			resources, exists := nodeResources[member.Name]
			if !exists {
				fmt.Printf("\tResource data not found for node %s\n", member.Name)
				continue
			}

			// Prepare the output string
			nodeInfo := fmt.Sprintf(
				"\tCoordinate: %+v\n\tRemaining CPU: %.2f%%\n\tRemaining Memory: %.2f MiB\n",
				coord, resources.RemainingCPU, resources.RemainingMemory,
			)

			// Print to console
			fmt.Print(nodeInfo)

			// Log the full data to file
			logger.Printf("Node Name: %s\nAddress: %s:%d\nStatus: %s\nTags: %v\nCoordinate: %+v\nRemaining CPU: %.2f%%\nRemaining Memory: %.2f MiB\n",
				member.Name, member.Addr, member.Port, member.Status, member.Tags, coord, resources.RemainingCPU, resources.RemainingMemory)
		}

		// Wait for a specified duration before the next iteration
		time.Sleep(10 * time.Second)
	}
}

// loadNodeResources reads the resource data from the given file and returns a map of NodeResources
func loadNodeResources(filePath string) (map[string]NodeResources, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	resources := make(map[string]NodeResources)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ", ")
		if len(parts) != 3 {
			continue // Skip lines that don't match the expected format
		}

		// Parse node name
		nameParts := strings.Split(parts[0], ": ")
		if len(nameParts) != 2 {
			continue
		}
		nodeName := strings.TrimSpace(nameParts[1])

		// Parse remaining CPU
		cpuParts := strings.Split(parts[1], ": ")
		if len(cpuParts) != 2 {
			continue
		}
		remainingCPU, err := strconv.ParseFloat(strings.TrimSuffix(strings.TrimSpace(cpuParts[1]), "%"), 64)
		if err != nil {
			continue
		}

		// Parse remaining memory
		memoryParts := strings.Split(parts[2], ": ")
		if len(memoryParts) != 2 {
			continue
		}
		memoryStr := strings.TrimSuffix(strings.TrimSpace(memoryParts[1]), " MiB")
		memoryStr = strings.TrimSuffix(memoryStr, " GiB") // Handle both MiB and GiB
		remainingMemory, err := strconv.ParseFloat(memoryStr, 64)
		if err != nil {
			continue
		}
		if strings.Contains(memoryParts[1], "GiB") {
			remainingMemory *= 1024 // Convert GiB to MiB
		}

		// Add to the map
		resources[nodeName] = NodeResources{
			RemainingCPU:    remainingCPU,
			RemainingMemory: remainingMemory,
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return resources, nil
}
