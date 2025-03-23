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

// Normalize a value to range [0, scaleMax]
func normalizeAndScale(value, min, max float64, scaleMax uint32) uint32 {
	if max == min {
		return 0 // Avoid division by zero if all values are the same
	}
	normalized := (value - min) / (max - min)
	return uint32(math.Round(normalized * float64(scaleMax)))
}

// Convert 4D (X, Y, Memory, CPU) to 1D Hilbert Index
func hilbert4D(order uint32, x, y, memory, cpu uint32) uint32 {
	sm, _ := hilbert.New(order, 4)
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(memory), uint64(cpu))
	return uint32(hilbertIndex.Uint64()) // Convert *big.Int to uint64, then cast to uint32
}

// Compute Hilbert Index for 4D data (Now properly normalizing all dimensions)
func ComputeHilbertValue(x, y float64, memory, cpu uint32, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) uint32 {
	hilbertOrder := uint32(10)
	scaleMax := uint32((1 << hilbertOrder) - 1)

	xInt := normalizeAndScale(x, minX, maxX, scaleMax)
	yInt := normalizeAndScale(y, minY, maxY, scaleMax)
	memInt := normalizeAndScale(float64(memory), minMem, maxMem, scaleMax)
	cpuInt := normalizeAndScale(float64(cpu), minCPU, maxCPU, scaleMax)

	return hilbert4D(hilbertOrder, xInt, yInt, memInt, cpuInt)
}

// Compute Hilbert indices and normalize values
func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64, float64, float64, float64, float64) {
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64
	minMem, maxMem := math.MaxFloat64, -math.MaxFloat64
	minCPU, maxCPU := math.MaxFloat64, -math.MaxFloat64

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
		if float64(node.Memory) < minMem {
			minMem = float64(node.Memory)
		}
		if float64(node.Memory) > maxMem {
			maxMem = float64(node.Memory)
		}
		if float64(node.CPU) < minCPU {
			minCPU = float64(node.CPU)
		}
		if float64(node.CPU) > maxCPU {
			maxCPU = float64(node.CPU)
		}
	}

	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].X, nodes[i].Y, nodes[i].Memory, nodes[i].CPU, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
	}

	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU
}

// Query using Hilbert Index
func QueryNodesUsingHilbertIndex(nodes []Node, queryMinX, queryMaxX, queryMinY, queryMaxY float64, queryMinMemory, queryMinCPU uint32, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) []Node {
	queryMinHilbert := ComputeHilbertValue(queryMinX, queryMinY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
	queryMaxHilbert := ComputeHilbertValue(queryMaxX, queryMaxY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)

	var result []Node
	for _, node := range nodes {
		if node.Hilbert1D >= queryMinHilbert && node.Hilbert1D <= queryMaxHilbert {
			result = append(result, node)
		}
	}
	return result
}

// Query using original data
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
	nodes := []Node{
		{-0.03183177785974009, -0.04379091279815393, 128, 48, 0},

		{-0.02028281011743071, -0.00017112118326381717, 16, 12, 0},

		{0.010579333353596287, 0.09237783955665406, 256, 124, 0},

		{0.034761437624688724, 0.011946346751751113, 8, 8, 0},
	}

	fmt.Println("Original Dataset:")
	for _, node := range nodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d\n", node.X, node.Y, node.Memory, node.CPU)
	}

	transformedNodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU := HilbertTransform(nodes)

	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}

	// Query parameters
	queryMinX := -0.02
	queryMaxX := 0.02
	queryMinY := -0.01
	queryMaxY := 0.1
	queryMinMemory := uint32(10)
	queryMinCPU := uint32(10)

	// Query using Hilbert Index
	queryResultsHilbert := QueryNodesUsingHilbertIndex(transformedNodes, queryMinX, queryMaxX, queryMinY, queryMaxY, queryMinMemory, queryMinCPU, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)

	// Query using original data
	queryResultsOriginal := QueryNodesUsingOriginalData(nodes, queryMinX, queryMaxX, queryMinY, queryMaxY, queryMinMemory, queryMinCPU)

	// Print results
	fmt.Println("\nQuery Results Using Hilbert Index:")
	for _, node := range queryResultsHilbert {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n", node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}

	fmt.Println("\nQuery Results Using Original Data:")
	for _, node := range queryResultsOriginal {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d\n", node.X, node.Y, node.Memory, node.CPU)
	}
}
