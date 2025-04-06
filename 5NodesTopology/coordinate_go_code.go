package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hashicorp/serf/client"
)

func main() {
	// Set up logging to a file for coordinates
	coordLogFile, err := os.OpenFile("coordinate.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open coordinate log file: %v", err)
	}
	defer coordLogFile.Close()
	coordLogger := log.New(coordLogFile, "", log.LstdFlags)

	// Set up the Serf RPC client configuration
	clientConfig := &client.Config{
		Addr: "127.0.0.1:7373", // Use localhost for RPC address
	}

	// Create a Serf RPC client
	serfClient, err := client.ClientFromConfig(clientConfig)
	if err != nil {
		log.Fatalf("Failed to create Serf client: %v", err)
	}
	defer serfClient.Close()

	for {
		// Retrieve and print members from the Serf client
		clientMembers, err := serfClient.Members()
		if err != nil {
			log.Fatalf("Failed to retrieve members from client: %v", err)
		}

		for _, member := range clientMembers {
			fmt.Printf("Node: %s, Address: %s:%d, Status: %s, Tags: %v\n",
				member.Name, member.Addr, member.Port, member.Status, member.Tags)

			// Fetch the network coordinate for the member
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil {
				fmt.Printf("Failed to get coordinate for node %s: %v\n", member.Name, err)
			} else {
				fmt.Printf("Vivaldi Coordinate for %s: %+v\n", member.Name, coord)

				// Log the coordinate to the coordinate log file with timestamp
				coordLogger.Printf("Time: %s - Coordinate for %s: %+v\n", time.Now().Format(time.RFC3339), member.Name, coord)
			}
		}

		// Wait for a specified duration (5 seconds) before the next coordinate retrieval
		time.Sleep(5 * time.Second) // Updated interval to 5 seconds
	}
}
