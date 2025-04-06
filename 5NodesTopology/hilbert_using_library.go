package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"

	"github.com/jtejido/hilbert"
)

const hilbertOrder = 16
const scaleMax = (1 << hilbertOrder) - 1

type Node struct {
	X         float64
	Y         float64
	Memory    uint32
	CPU       uint32
	Hilbert1D uint64
}

func normalizeAndScale(value, min, max float64) uint32 {
	if max == min {
		return 0
	}
	normalized := (value - min) / (max - min)
	// Clamp normalized to [0, 1]
	normalized = math.Max(0, math.Min(1, normalized))
	return uint32(math.Round(normalized * float64(scaleMax)))
}

func denormalize(value uint32, min, max float64) float64 {
	if scaleMax == 0 {
		return min
	}
	normalized := float64(value) / float64(scaleMax)
	return min + normalized*(max-min)
}

func hilbert4D(x, y, memory, cpu uint32) uint64 {
	sm, _ := hilbert.New(hilbertOrder, 4)
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(memory), uint64(cpu))
	return hilbertIndex.Uint64()
}

func decodeHilbert4D(hilbertValue uint64) (uint32, uint32, uint32, uint32) {
	sm, _ := hilbert.New(hilbertOrder, 4)
	coords := sm.Decode(new(big.Int).SetUint64(hilbertValue))
	return uint32(coords[0]), uint32(coords[1]), uint32(coords[2]), uint32(coords[3])
}

func ComputeHilbertValue(x, y float64, memory, cpu uint32, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) uint64 {
	xInt := normalizeAndScale(x, minX, maxX)
	yInt := normalizeAndScale(y, minY, maxY)
	memInt := normalizeAndScale(float64(memory), minMem, maxMem)
	cpuInt := normalizeAndScale(float64(cpu), minCPU, maxCPU)

	return hilbert4D(xInt, yInt, memInt, cpuInt)
}

func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64, float64, float64, float64, float64) {
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64
	minMem, maxMem := math.MaxFloat64, -math.MaxFloat64
	minCPU, maxCPU := math.MaxFloat64, -math.MaxFloat64

	// Find min/max for normalization
	for _, node := range nodes {
		minX = math.Min(minX, node.X)
		maxX = math.Max(maxX, node.X)
		minY = math.Min(minY, node.Y)
		maxY = math.Max(maxY, node.Y)
		minMem = math.Min(minMem, float64(node.Memory))
		maxMem = math.Max(maxMem, float64(node.Memory))
		minCPU = math.Min(minCPU, float64(node.CPU))
		maxCPU = math.Max(maxCPU, float64(node.CPU))
	}

	// Assign Hilbert value
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(
			nodes[i].X, nodes[i].Y, nodes[i].Memory, nodes[i].CPU,
			minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
	}

	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU
}

func DecodeHilbertValue(hilbertVal uint64, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) (float64, float64, float64, float64) {
	xInt, yInt, memInt, cpuInt := decodeHilbert4D(hilbertVal)

	x := denormalize(xInt, minX, maxX)
	y := denormalize(yInt, minY, maxY)
	mem := denormalize(memInt, minMem, maxMem)
	cpu := denormalize(cpuInt, minCPU, maxCPU)

	return x, y, mem, cpu
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
		fmt.Printf("X: %.6f, Y: %.6f, Memory: %d, CPU: %d\n", node.X, node.Y, node.Memory, node.CPU)
	}

	transformedNodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU := HilbertTransform(nodes)

	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %.6f, Y: %.6f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}

	fmt.Println("\nDecoded values from Hilbert1D:")
	for _, node := range transformedNodes {
		x, y, mem, cpu := DecodeHilbertValue(node.Hilbert1D, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
		fmt.Printf("Hilbert1D: %d => X: %.6f, Y: %.6f, Memory: %.2f, CPU: %.2f\n", node.Hilbert1D, x, y, mem, cpu)
	}
}
