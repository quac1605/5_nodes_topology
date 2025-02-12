package main

import (
	"fmt"
	"sort"

	"github.com/jtejido/hilbert"
)

// Node represents a network node with 3D attributes
type Node struct {
	RTT       uint32
	Memory    uint32
	CPU       uint32
	Hilbert1D uint32
}

// Convert 3D (RTT, Memory, CPU) to 1D Hilbert Index
func hilbert3D(order uint32, x, y, z uint32) uint32 {
	// Create a new Hilbert space mapper with the specified order and 3D space
	sm, _ := hilbert.New(order, 3)

	// Encode the x, y, z values, but cast them to uint64
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(z))

	// Convert the hilbertIndex (which is of type *big.Int) to uint32
	hilbertIndexUint32 := uint32(hilbertIndex.Uint64()) // Convert *big.Int to uint64, then cast to uint32

	// Return the Hilbert index
	return hilbertIndexUint32
}

// Compute Hilbert Index after normalizing
func ComputeHilbertValue(rtt, memory, cpu uint32) uint32 {
	// Use Hilbert order of 10 (fits 3D space properly for [0, 1023])
	hilbertOrder := uint32(8)

	// Calculate the Hilbert index
	return hilbert3D(hilbertOrder, rtt, memory, cpu)
}

// HilbertTransform computes and sorts nodes by Hilbert 1D value
func HilbertTransform(nodes []Node) []Node {
	// Normalize and calculate Hilbert 1D for each node
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].RTT, nodes[i].Memory, nodes[i].CPU)
	}

	// Sort nodes based on Hilbert 1D index
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes
}

// QueryNodes finds nodes matching constraints
func QueryNodes(nodes []Node, maxRTT, minMemory, minCPU uint32) []Node {
	var result []Node
	for _, node := range nodes {
		if node.RTT <= maxRTT && node.Memory >= minMemory && node.CPU >= minCPU {
			result = append(result, node)
		}
	}
	return result
}

func main() {
	// Sample dataset (RTT, Memory, CPU)
	nodes := []Node{
		{100, 128, 48, 0},
		{50, 16, 12, 0},
		{100, 256, 124, 0},
		{50, 8, 8, 0},
		{20, 32, 16, 0},
	}

	// Print the original dataset
	fmt.Println("Original Dataset:")
	for _, node := range nodes {
		fmt.Printf("RTT: %d, Memory: %d, CPU: %d\n", node.RTT, node.Memory, node.CPU)
	}

	// Convert dataset to 1D Hilbert space
	transformedNodes := HilbertTransform(nodes)

	// Print the transformed dataset with Hilbert 1D values
	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("RTT: %d, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.RTT, node.Memory, node.CPU, node.Hilbert1D)
	}

	// Query parameters
	maxRTT := uint32(100)
	minMemory := uint32(40)
	minCPU := uint32(6)

	// Print query command
	fmt.Printf("\nQuery Command: Find nodes with RTT ≤ %dms, Memory ≥ %dMB, CPU ≥ %d cores\n", maxRTT, minMemory, minCPU)

	// Perform query
	queryResults := QueryNodes(transformedNodes, maxRTT, minMemory, minCPU)

	// Print query results
	fmt.Println("\nQuery Results:")
	for _, node := range queryResults {
		fmt.Printf("RTT: %d, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.RTT, node.Memory, node.CPU, node.Hilbert1D)
	}
}
