package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/jtejido/hilbert"
)

// Node represents a network node with 4D attributes
type Node struct {
	X         float64
	Y         float64
	Memory    uint32
	CPU       uint32
	Hilbert1D uint32
}

// Normalize and scale floating-point coordinates to uint32 range
func normalizeAndScale(value, min, max float64, scaleMax uint32) uint32 {
	// Normalize to [0, 1]
	normalized := (value - min) / (max - min)

	// Scale to uint32 range [0, scaleMax]
	return uint32(math.Round(normalized * float64(scaleMax)))
}

// Convert 4D (X, Y, Memory, CPU) to 1D Hilbert Index
func hilbert4D(order uint32, x, y, memory, cpu uint32) uint32 {
	// Create a new Hilbert space mapper with the specified order and 4D space
	sm, _ := hilbert.New(order, 4)

	// Encode the x, y, memory, cpu values
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(memory), uint64(cpu))

	// Convert the hilbertIndex (which is of type *big.Int) to uint32
	return uint32(hilbertIndex.Uint64()) // Convert *big.Int to uint64, then cast to uint32
}

// Compute Hilbert Index for 4D data with floating point normalization
func ComputeHilbertValue(x, y float64, memory, cpu uint32, minX, maxX, minY, maxY float64) uint32 {
	hilbertOrder := uint32(8)
	scaleMax := uint32((1 << hilbertOrder) - 1) // 2^order - 1

	// Normalize and scale X and Y to uint32
	xInt := normalizeAndScale(x, minX, maxX, scaleMax)
	yInt := normalizeAndScale(y, minY, maxY, scaleMax)

	// Calculate Hilbert index
	return hilbert4D(hilbertOrder, xInt, yInt, memory, cpu)
}

// HilbertTransform computes and sorts nodes by Hilbert 1D value
func HilbertTransform(nodes []Node) []Node {
	// Find min/max for X and Y for normalization
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64

	for _, node := range nodes {
		if node.X < minX {
			minX = node.X
		}
		if node.X > maxX {
			maxX = node.X
		}
		if node.Y < minY {
			minY = node.Y
		}
		if node.Y > maxY {
			maxY = node.Y
		}
	}

	// Normalize and compute Hilbert 1D for each node
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].X, nodes[i].Y, nodes[i].Memory, nodes[i].CPU, minX, maxX, minY, maxY)
	}

	// Sort nodes based on Hilbert 1D index
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes
}

// QueryNodes finds nodes matching constraints
func QueryNodes(nodes []Node, minX, maxX, minY, maxY float64, minMemory, minCPU uint32) []Node {
	var result []Node
	for _, node := range nodes {
		if node.X >= minX && node.X <= maxX &&
			node.Y >= minY && node.Y <= maxY &&
			node.Memory >= minMemory &&
			node.CPU >= minCPU {
			result = append(result, node)
		}
	}
	return result
}

func main() {
	// Sample dataset (X, Y in seconds, Memory, CPU)
	nodes := []Node{
		{-0.03183177785974009, -0.04379091279815393, 128, 48, 0},
		{-0.02028281011743071, -0.00017112118326381717, 16, 12, 0},
		{0.010579333353596287, 0.09237783955665406, 256, 124, 0},
		{0.034761437624688724, 0.011946346751751113, 8, 8, 0},
	}

	// Print the original dataset
	fmt.Println("Original Dataset:")
	for _, node := range nodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d\n", node.X, node.Y, node.Memory, node.CPU)
	}

	// Convert dataset to 1D Hilbert space
	transformedNodes := HilbertTransform(nodes)

	// Print the transformed dataset with Hilbert 1D values
	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}
	// Query parameters
	queryMinX := -0.02      // X ≥ -0.02
	queryMaxX := 0.02       // X ≤ 0.02
	queryMinY := -0.01      // Y ≥ -0.01
	queryMaxY := 0.1        // Y ≤ 0.05
	minMemory := uint32(10) // Memory ≥ 10MB
	minCPU := uint32(10)    // CPU ≥ 10 cores

	// Print query command
	fmt.Printf("\nQuery Command: Find nodes with X ∈ [%.2f, %.2f], Y ∈ [%.2f, %.2f], Memory ≥ %dMB, CPU ≥ %d cores\n",
		queryMinX, queryMaxX, queryMinY, queryMaxY, minMemory, minCPU)

	// Perform query
	queryResults := QueryNodes(nodes, queryMinX, queryMaxX, queryMinY, queryMaxY, minMemory, minCPU)

	// Print query results
	fmt.Println("\nQuery Results:")
	if len(queryResults) == 0 {
		fmt.Println("No nodes found matching the criteria.")
	} else {
		for _, node := range queryResults {
			fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
				node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
		}
	}
}
