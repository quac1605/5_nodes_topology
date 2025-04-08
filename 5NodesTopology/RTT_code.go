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

	logFile, err := os.OpenFile("rtt_estimates.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {

		log.Fatalf("Failed to open log file: %v", err)

	}

	defer logFile.Close()

	logger := log.New(logFile, "", log.LstdFlags)

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

				// Calculate RTT from coordinates to this node

				for _, otherMember := range clientMembers {

					if member.Name != otherMember.Name {

						otherCoord, err := serfClient.GetCoordinate(otherMember.Name)

						if err != nil {

							fmt.Printf("Failed to get coordinate for node %s: %v\n", otherMember.Name, err)

						} else {

							rtt := calculateRTT(coord, otherCoord)

							fmt.Printf("Estimated RTT from %s to %s: %.2f ms\n", member.Name, otherMember.Name, rtt)

							// Log the RTT to the file with timestamp

							logger.Printf("Time: %s - Estimated RTT from %s to %s: %.2f ms\n", time.Now().Format(time.RFC3339), member.Name, otherMember.Name, rtt)

						}

					}

				}

			}

		}

		// Wait for a specified duration before next RTT calculation

		time.Sleep(10 * time.Second) // Adjust the interval as needed

	}

}

// calculateRTT calculates the RTT between two Vivaldi coordinates

func calculateRTT(a, b *coordinate.Coordinate) float64 {

	// Coordinates will always have the same dimensionality, so this is

	// just a sanity check.

	if len(a.Vec) != len(b.Vec) {

		panic("dimensions aren't compatible")

	}

	// Calculate the Euclidean distance plus the heights.

	sumsq := 0.0

	for i := 0; i < len(a.Vec); i++ {

		diff := a.Vec[i] - b.Vec[i]

		sumsq += diff * diff

	}

	rtt := math.Sqrt(sumsq) + a.Height + b.Height

	// Apply the adjustment components, guarding against negatives.

	adjusted := rtt + a.Adjustment + b.Adjustment

	if adjusted > 0.0 {

		rtt = adjusted

	}

	return rtt * 1000 // Convert to milliseconds

}
