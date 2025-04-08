package main

import (
	"fmt"
	"log"
	"math"
	"math/big"
	"sort"

	"github.com/hashicorp/serf/client"
	"github.com/jtejido/hilbert"
)

const hilbertOrder = 16
const scaleMax = (1 << hilbertOrder) - 1

type Node struct {
	Name      string
	X         float64
	Y         float64
	Hilbert1D uint64
}

// Normalize and scale coordinates to a range between 0 and scaleMax
func normalizeAndScale(value, min, max float64) uint32 {
	if max == min {
		return 0
	}
	normalized := (value - min) / (max - min)
	// Clamp normalized value to [0, 1]
	normalized = math.Max(0, math.Min(1, normalized))
	return uint32(math.Round(normalized * float64(scaleMax)))
}

// Denormalize the value to its original range
func denormalize(value uint32, min, max float64) float64 {
	if scaleMax == 0 {
		return min
	}
	normalized := float64(value) / float64(scaleMax)
	return min + normalized*(max-min)
}

// 2D Hilbert transform for X and Y coordinates
func hilbert2D(x, y uint32) uint64 {
	sm, _ := hilbert.New(hilbertOrder, 2) // 2D Hilbert curve
	hilbertIndex := sm.Encode(uint64(x), uint64(y))
	return hilbertIndex.Uint64()
}

// Decode the Hilbert value back into 2D coordinates
func decodeHilbert2D(hilbertValue uint64) (uint32, uint32) {
	sm, _ := hilbert.New(hilbertOrder, 2) // 2D Hilbert curve
	coords := sm.Decode(new(big.Int).SetUint64(hilbertValue))
	return uint32(coords[0]), uint32(coords[1])
}

// Compute Hilbert 1D value based on normalized X, Y coordinates
func ComputeHilbertValue(x, y float64, minX, maxX, minY, maxY float64) uint64 {
	xInt := normalizeAndScale(x, minX, maxX)
	yInt := normalizeAndScale(y, minY, maxY)
	return hilbert2D(xInt, yInt)
}

// Apply Hilbert Transform to a set of nodes
func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64) {
	// Initialize the min and max values for X and Y
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64

	// Find min/max for normalization
	for _, node := range nodes {
		minX = math.Min(minX, node.X)
		maxX = math.Max(maxX, node.X)
		minY = math.Min(minY, node.Y)
		maxY = math.Max(maxY, node.Y)
	}

	// Assign Hilbert value based on the normalized coordinates
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(
			nodes[i].X, nodes[i].Y, minX, maxX, minY, maxY)
	}

	// Sort nodes by their Hilbert 1D value
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes, minX, maxX, minY, maxY
}

// Decode Hilbert 1D value to original X and Y coordinates
func DecodeHilbertValue(hilbertVal uint64, minX, maxX, minY, maxY float64) (float64, float64) {
	xInt, yInt := decodeHilbert2D(hilbertVal)
	x := denormalize(xInt, minX, maxX)
	y := denormalize(yInt, minY, maxY)
	return x, y
}

func main() {
	// Connect to Serf client (example: change the address to your Serf RPC address)
	serfClient, err := client.NewRPCClient("127.0.0.1:7373")
	if err != nil {
		log.Fatalf("Failed to connect to Serf agent: %v", err)
	}
	defer serfClient.Close()

	// Get members from Serf cluster
	serfMembers, err := serfClient.Members()
	if err != nil {
		log.Fatalf("Failed to get members: %v", err)
	}

	var nodes []Node
	// Iterate through Serf members and extract coordinates
	for _, member := range serfMembers {
		// Only process nodes that have coordinates assigned
		coord, err := serfClient.GetCoordinate(member.Name)
		if err != nil || coord == nil {
			log.Printf("Warning: Coordinate unavailable for %s, skipping...", member.Name)
			continue
		}
		nodes = append(nodes, Node{
			Name: member.Name,
			X:    coord.Vec[0], // X coordinate
			Y:    coord.Vec[1], // Y coordinate
		})
	}

	// Print the original data with coordinates
	fmt.Println("Original Data with Coordinates:")
	for _, node := range nodes {
		fmt.Printf("%s -> X: %.6f, Y: %.6f\n", node.Name, node.X, node.Y)
	}

	// Apply the Hilbert Transform
	transformedNodes, minX, maxX, minY, maxY := HilbertTransform(nodes)

	// Print the sorted nodes by Hilbert 1D values
	fmt.Println("\nSorted Nodes by Hilbert 1D Value:")
	for _, node := range transformedNodes {
		fmt.Printf("%s => Hilbert1D: %d\n", node.Name, node.Hilbert1D)
	}

	// Decode Hilbert values back to X, Y coordinates
	fmt.Println("\nDecoded from Hilbert 1D:")
	for _, node := range transformedNodes {
		x, y := DecodeHilbertValue(node.Hilbert1D, minX, maxX, minY, maxY)
		fmt.Printf("%s => X: %.6f, Y: %.6f\n", node.Name, x, y)
	}
}
