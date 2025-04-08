package main

import (
	"fmt"
	"log"
	"os"
	"sort"
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
		clientMembers, err := serfClient.Members()
		if err != nil {
			log.Fatalf("Failed to retrieve members from client: %v", err)
		}

		coordinates := make(map[string]string)
		var nodeNames []string

		for _, member := range clientMembers {
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil || coord == nil {
				coordinates[member.Name] = "Unavailable"
			} else {
				coordinates[member.Name] = fmt.Sprintf("%v", coord.Vec)
			}
			nodeNames = append(nodeNames, member.Name)
		}

		sort.Strings(nodeNames)

		timestamp := time.Now().Format(time.RFC3339)
		fmt.Printf("\n--- Coordinate Snapshot @ %s ---\n", timestamp)
		coordLogger.Printf("--- Coordinate Snapshot @ %s ---", timestamp)

		for _, name := range nodeNames {
			vec := coordinates[name]
			fmt.Printf("%s: %s\n", name, vec)
			coordLogger.Printf("%s: %s\n", name, vec)
		}

		time.Sleep(5 * time.Second)
	}
}
