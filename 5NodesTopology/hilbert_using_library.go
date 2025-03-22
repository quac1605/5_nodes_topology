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
	normalized := (value - min) / (max - min)                 // Normalize to [0,1]
	return uint32(math.Round(normalized * float64(scaleMax))) // Scale to [0, scaleMax]
}

// Convert 4D (X, Y, Memory, CPU) to 1D Hilbert Index
func hilbert4D(order uint32, x, y, memory, cpu uint32) uint32 {
	sm, _ := hilbert.New(order, 4)
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(memory), uint64(cpu))
	return uint32(hilbertIndex.Uint64()) // Convert *big.Int to uint64, then cast to uint32
}

// Compute Hilbert Index for 4D data
func ComputeHilbertValue(x, y float64, memory, cpu uint32, minX, maxX, minY, maxY float64) uint32 {
	hilbertOrder := uint32(10)
	scaleMax := uint32((1 << hilbertOrder) - 1) // 2^order - 1

	// Normalize and scale X and Y to uint32
	xInt := normalizeAndScale(x, minX, maxX, scaleMax)
	yInt := normalizeAndScale(y, minY, maxY, scaleMax)

	// Compute Hilbert index
	return hilbert4D(hilbertOrder, xInt, yInt, memory, cpu)
}

// HilbertTransform computes and sorts nodes by Hilbert 1D value
func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64) {
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

	return nodes, minX, maxX, minY, maxY
}

// QueryNodesUsingHilbertIndex filters by Hilbert index first, then applies original constraints
func QueryNodesUsingHilbertIndex(nodes []Node, queryMinX, queryMaxX, queryMinY, queryMaxY float64, queryMinMemory, queryMinCPU uint32, minX, maxX, minY, maxY float64) []Node {
	// Compute Hilbert index bounds for the query range
	queryMinHilbert := ComputeHilbertValue(queryMinX, queryMinY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY)
	queryMaxHilbert := ComputeHilbertValue(queryMaxX, queryMaxY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY)

	fmt.Printf("\nQuery Hilbert Index Range: [%d, %d]\n", queryMinHilbert, queryMaxHilbert)
	// Ensure the query range is valid
	if queryMinHilbert > queryMaxHilbert {
		return nil
	}

	// Find nodes within the Hilbert index range
	var result []Node
	for _, node := range nodes {
		if node.Hilbert1D >= queryMinHilbert && node.Hilbert1D <= queryMaxHilbert {
			result = append(result, node)
		}
	}

	return result
}

// QueryNodesUsingOriginalData searches for nodes using original attribute constraints
func QueryNodesUsingOriginalData(nodes []Node, queryMinX, queryMaxX, queryMinY, queryMaxY float64, queryMinMemory, queryMinCPU uint32) []Node {
	var result []Node
	for _, node := range nodes {
		if node.X >= queryMinX && node.X <= queryMaxX &&
			node.Y >= queryMinY && node.Y <= queryMaxY &&
			node.Memory >= queryMinMemory &&
			node.CPU >= queryMinCPU {
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
	transformedNodes, minX, maxX, minY, maxY := HilbertTransform(nodes)

	// Print the transformed dataset with Hilbert 1D values
	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}

	// Query parameters
	queryMinX := -0.02           // X ≥ -0.02
	queryMaxX := 0.02            // X ≤ 0.02
	queryMinY := -0.01           // Y ≥ -0.01
	queryMaxY := 0.1             // Y ≤ 0.1
	queryMinMemory := uint32(10) // Memory ≥ 10MB
	queryMinCPU := uint32(10)    // CPU ≥ 10 cores

	// Print query command
	fmt.Printf("\nQuery Command: Find nodes with X ∈ [%.2f, %.2f], Y ∈ [%.2f, %.2f], Memory ≥ %dMB, CPU ≥ %d cores\n",
		queryMinX, queryMaxX, queryMinY, queryMaxY, queryMinMemory, queryMinCPU)

	// Perform query using Hilbert 1D Index
	queryResultsHilbert := QueryNodesUsingHilbertIndex(transformedNodes, queryMinX, queryMaxX, queryMinY, queryMaxY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY)

	// Print query results using Hilbert Index
	fmt.Println("\nQuery Results Using Hilbert Index:")
	if len(queryResultsHilbert) == 0 {
		fmt.Println("No nodes found matching the criteria.")
	} else {
		for _, node := range queryResultsHilbert {
			fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
				node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
		}
	}

	// Perform query using original data
	queryResultsOriginal := QueryNodesUsingOriginalData(nodes, queryMinX, queryMaxX, queryMinY, queryMaxY, queryMinMemory, queryMinCPU)

	// Print query results using original data
	fmt.Println("\nQuery Results Using Original Data:")
	if len(queryResultsOriginal) == 0 {
		fmt.Println("No nodes found matching the criteria.")
	} else {
		for _, node := range queryResultsOriginal {
			fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d\n",
				node.X, node.Y, node.Memory, node.CPU)
		}
	}

	// Compare the results
	if len(queryResultsHilbert) == len(queryResultsOriginal) {
		fmt.Println("\nBoth methods returned the same number of results.")
	} else {
		fmt.Println("\nThe methods returned different numbers of results.")
	}
}
