package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"

	"github.com/jtejido/hilbert"
)

// Node represents a network node with 2D attributes
type Node struct {
	X         float64
	Y         float64
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

// Denormalize a value back to the original range
func denormalize(value uint32, min, max float64, scaleMax uint32) float64 {
	return min + (float64(value)/float64(scaleMax))*(max-min)
}

// Convert 2D (X, Y) to 1D Hilbert Index
func hilbert2D(order uint32, x, y uint32) uint32 {
	sm, _ := hilbert.New(order, 2)
	hilbertIndex := sm.Encode(uint64(x), uint64(y))
	return uint32(hilbertIndex.Uint64()) // Convert *big.Int to uint64, then cast to uint32
}

// Decode Hilbert Index back to (X, Y) coordinates
func DecodeHilbertValue(hilbertValue uint32, minX, maxX, minY, maxY float64) (float64, float64) {
	hilbertOrder := uint32(10)
	scaleMax := uint32((1 << hilbertOrder) - 1)

	sm, _ := hilbert.New(hilbertOrder, 2)

	// Convert uint32 to *big.Int
	hilbertBigInt := new(big.Int).SetUint64(uint64(hilbertValue))

	// Decode Hilbert Index (returns a slice)
	decoded := sm.Decode(hilbertBigInt)

	// Extract X and Y from slice
	xInt, yInt := uint32(decoded[0]), uint32(decoded[1])

	// Convert back to original float range
	x := denormalize(xInt, minX, maxX, scaleMax)
	y := denormalize(yInt, minY, maxY, scaleMax)

	return x, y
}

// Compute Hilbert Index for 2D data
func ComputeHilbertValue(x, y float64, minX, maxX, minY, maxY float64) uint32 {
	hilbertOrder := uint32(10)
	scaleMax := uint32((1 << hilbertOrder) - 1)

	xInt := normalizeAndScale(x, minX, maxX, scaleMax)
	yInt := normalizeAndScale(y, minY, maxY, scaleMax)

	return hilbert2D(hilbertOrder, xInt, yInt)
}

// Compute Hilbert indices and normalize values
func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64) {
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

	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].X, nodes[i].Y, minX, maxX, minY, maxY)
	}

	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes, minX, maxX, minY, maxY
}

// Query using Hilbert Index
func QueryNodesUsingHilbertIndex(nodes []Node, queryMinX, queryMaxX, queryMinY, queryMaxY float64, minX, maxX, minY, maxY float64) []Node {
	queryMinHilbert := ComputeHilbertValue(queryMinX, queryMinY, minX, maxX, minY, maxY)
	queryMaxHilbert := ComputeHilbertValue(queryMaxX, queryMaxY, minX, maxX, minY, maxY)
	fmt.Printf("Query Hilbert1D Range: [%d, %d]\n", queryMinHilbert, queryMaxHilbert)
	var result []Node
	for _, node := range nodes {
		if node.Hilbert1D >= queryMinHilbert && node.Hilbert1D <= queryMaxHilbert {
			result = append(result, node)
		}
	}
	return result
}

// Query using original data
func QueryNodesUsingOriginalData(nodes []Node, queryMinX, queryMaxX, queryMinY, queryMaxY float64) []Node {
	var result []Node
	for _, node := range nodes {
		if node.X >= queryMinX && node.X <= queryMaxX &&
			node.Y >= queryMinY && node.Y <= queryMaxY {
			result = append(result, node)
		}
	}
	return result
}

func main() {
	nodes := []Node{
		{-0.03183177785974009, -0.04379091279815393, 0},
		{-0.02028281011743071, -0.00017112118326381717, 0},
		{0.010579333353596287, 0.09237783955665406, 0},
		{0.034761437624688724, 0.011946346751751113, 0},
	}

	fmt.Println("Original Dataset:")
	for _, node := range nodes {
		fmt.Printf("X: %f, Y: %f\n", node.X, node.Y)
	}

	transformedNodes, minX, maxX, minY, maxY := HilbertTransform(nodes)

	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %f, Y: %f => Hilbert1D: %d\n", node.X, node.Y, node.Hilbert1D)
	}

	// Decode Hilbert Index back to (X, Y)
	fmt.Println("\nDecoded Hilbert Values Back to Coordinates:")
	for _, node := range nodes {
		decodedX, decodedY := DecodeHilbertValue(node.Hilbert1D, minX, maxX, minY, maxY)
		fmt.Printf("Hilbert1D: %d => Decoded X: %f, Decoded Y: %f\n", node.Hilbert1D, decodedX, decodedY)
	}

	// Query parameters
	queryMinX := -0.02
	queryMaxX := 0.02
	queryMinY := -0.01
	queryMaxY := 0.1

	// Query using Hilbert Index
	queryResultsHilbert := QueryNodesUsingHilbertIndex(transformedNodes, queryMinX, queryMaxX, queryMinY, queryMaxY, minX, maxX, minY, maxY)

	// Query using original data
	queryResultsOriginal := QueryNodesUsingOriginalData(nodes, queryMinX, queryMaxX, queryMinY, queryMaxY)

	// Print results
	fmt.Println("\nQuery Results Using Hilbert Index:")
	for _, node := range queryResultsHilbert {
		fmt.Printf("X: %f, Y: %f => Hilbert1D: %d\n", node.X, node.Y, node.Hilbert1D)
	}

	fmt.Println("\nQuery Results Using Original Data:")
	for _, node := range queryResultsOriginal {
		fmt.Printf("X: %f, Y: %f\n", node.X, node.Y)
	}
}
