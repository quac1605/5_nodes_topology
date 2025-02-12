package main

import (
	"fmt"
	"math"
	"sort"
)

// Node represents a network node with 3D attributes
type Node struct {
	RTT       uint32
	Memory    uint32
	CPU       uint32
	Hilbert1D uint32
}

// Rotate and Flip Function for Hilbert Curve
func rot(n uint32, x, y, z *uint32, rx, ry, rz uint32) {
	if rz == 1 {
		*x, *y = *y, *x // Swap x and y
	}
	if ry == 0 {
		if rx == 1 {
			*x = n - 1 - *x
			*y = n - 1 - *y
		}
		*x, *z = *z, *x // Swap x and z
	}
}

// Convert 3D (RTT, Memory, CPU) to 1D Hilbert Index
func hilbert3D(order uint32, x, y, z uint32) uint32 {
	n := uint32(1 << order) // Grid size
	hilbertIndex := uint32(0)
	s := n >> 1

	for s > 0 {
		rx := (x & s) >> (order - 1)
		ry := (y & s) >> (order - 1)
		rz := (z & s) >> (order - 1)

		hilbertIndex += s * s * ((3 * rx) ^ ry)

		rot(s, &x, &y, &z, rx, ry, rz)
		s >>= 1
	}
	return hilbertIndex
}

// Normalize values to fit in the range [0, 1023]
func normalize(value, maxValue uint32) uint32 {
	return uint32(math.Round(float64(value) * 1023 / float64(maxValue)))
}

// Compute Hilbert Index after normalizing
func ComputeHilbertValue(rtt, memory, cpu uint32, maxRTT, maxMemory, maxCPU uint32) uint32 {
	// Normalize values to fit [0, 1023]
	normalizedRTT := normalize(rtt, maxRTT)
	normalizedMemory := normalize(memory, maxMemory)
	normalizedCPU := normalize(cpu, maxCPU)

	// Use Hilbert order of 10 (fits 3D space properly for [0, 1023])
	hilbertOrder := uint32(10)

	// Print the normalized values (for debugging)
	fmt.Printf("Normalized Values - RTT: %d, Memory: %d, CPU: %d\n", normalizedRTT, normalizedMemory, normalizedCPU)

	// Calculate the Hilbert index
	return hilbert3D(hilbertOrder, normalizedRTT, normalizedMemory, normalizedCPU)
}

// HilbertTransform computes and sorts nodes by Hilbert 1D value
func HilbertTransform(nodes []Node) []Node {
	// Find the max values in the dataset for proper normalization
	var maxRTT, maxMemory, maxCPU uint32
	for _, node := range nodes {
		if node.RTT > maxRTT {
			maxRTT = node.RTT
		}
		if node.Memory > maxMemory {
			maxMemory = node.Memory
		}
		if node.CPU > maxCPU {
			maxCPU = node.CPU
		}
	}

	// Normalize and calculate Hilbert 1D for each node
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].RTT, nodes[i].Memory, nodes[i].CPU, maxRTT, maxMemory, maxCPU)
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

	// Convert dataset to 1D Hilbert space
	transformedNodes := HilbertTransform(nodes)

	// Query parameters
	maxRTT := uint32(100)
	minMemory := uint32(16)
	minCPU := uint32(6)

	// Print query command
	fmt.Printf("Query Command: Find nodes with RTT ≤ %dms, Memory ≥ %dMB, CPU ≥ %d cores\n", maxRTT, minMemory, minCPU)

	// Perform query
	queryResults := QueryNodes(transformedNodes, maxRTT, minMemory, minCPU)

	// Print results
	fmt.Println("Query Results:")
	for _, node := range queryResults {
		fmt.Printf("RTT: %d, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.RTT, node.Memory, node.CPU, node.Hilbert1D)
	}
}
