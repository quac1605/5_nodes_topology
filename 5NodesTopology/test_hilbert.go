package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
)

func main() {
	// Set up logging to a file
	logFile, err := os.OpenFile("node_coordinates_with_hilbert.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
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
			// Print node information
			fmt.Printf("Node: %s, Address: %s:%d, Status: %s, Tags: %v\n",
				member.Name, member.Addr, member.Port, member.Status, member.Tags)

			// Fetch the network coordinate for the member
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil {
				fmt.Printf("Failed to get coordinate for node %s: %v\n", member.Name, err)
				continue
			}

			// Calculate Hilbert index using only coordinates
			hilbertIndex := calculateHilbertIndex(coord)

			// Prepare the output string
			nodeInfo := fmt.Sprintf(
				"Node: %s\n\tCoordinate: %+v\n\tHilbert Index: %d\n",
				member.Name, coord, hilbertIndex,
			)

			// Print to console
			fmt.Print(nodeInfo)

			// Log to file
			logger.Printf(nodeInfo)
		}

		// Wait for a specified duration before the next iteration
		time.Sleep(10 * time.Second)
	}
}

// calculateHilbertIndex calculates the Hilbert index for a node based on its coordinates
func calculateHilbertIndex(coord *coordinate.Coordinate) uint64 {
	// Define scaling factors
	scaleFactor := 1000.0

	// Normalize values
	x := int(math.Round(coord.Vec[0] * scaleFactor))
	y := int(math.Round(coord.Vec[1] * scaleFactor))

	// Calculate Hilbert index (2D)
	hilbertOrder := 10 // Number of bits per dimension
	return hilbertIndex2D(hilbertOrder, x, y)
}

// hilbertIndex2D encodes 2D coordinates into a Hilbert curve index
func hilbertIndex2D(order, x, y int) uint64 {
	var index uint64

	for s := order - 1; s >= 0; s-- {
		mask := 1 << s
		rx := (x & mask) >> s
		ry := (y & mask) >> s

		// Combine the bits into a Hilbert index
		index = (index << 2) | (uint64(rx)<<1 | uint64(ry))
	}
	return index
}
