package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"

	"github.com/jtejido/hilbert"
)

type Node struct {
	X         float64
	Y         float64
	Memory    uint32
	CPU       uint32
	Hilbert1D uint64 // <-- Change here
}

const hilbertOrder = uint32(16)
const scaleMax = uint32((1 << hilbertOrder) - 1)

func normalizeAndScale(value, min, max float64, scaleMax uint32) uint32 {
	if max == min {
		return 0
	}
	normalized := (value - min) / (max - min)
	return uint32(math.Round(normalized * float64(scaleMax)))
}

func denormalize(value uint32, min, max float64, scaleMax uint32) float64 {
	if scaleMax == 0 {
		return min
	}
	normalized := float64(value) / float64(scaleMax)
	return min + normalized*(max-min)
}

func hilbert4D(x, y, memory, cpu uint32) uint64 {
	sm, _ := hilbert.New(hilbertOrder, 4)
	hilbertIndex := sm.Encode(uint64(x), uint64(y), uint64(memory), uint64(cpu))
	return hilbertIndex.Uint64() // <-- No more truncating
}

func decodeHilbert4D(hilbertValue uint64) (uint32, uint32, uint32, uint32) {
	sm, _ := hilbert.New(hilbertOrder, 4)
	coords := sm.Decode(new(big.Int).SetUint64(hilbertValue))
	return uint32(coords[0]), uint32(coords[1]), uint32(coords[2]), uint32(coords[3])
}

func ComputeHilbertValue(x, y float64, memory, cpu uint32, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) uint64 {
	xInt := normalizeAndScale(x, minX, maxX, scaleMax)
	yInt := normalizeAndScale(y, minY, maxY, scaleMax)
	memInt := normalizeAndScale(float64(memory), minMem, maxMem, scaleMax)
	cpuInt := normalizeAndScale(float64(cpu), minCPU, maxCPU, scaleMax)

	return hilbert4D(xInt, yInt, memInt, cpuInt)
}

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
		nodes[i].Hilbert1D = ComputeHilbertValue(
			nodes[i].X, nodes[i].Y, nodes[i].Memory, nodes[i].CPU,
			minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU,
		)
	}

	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Hilbert1D < nodes[j].Hilbert1D
	})

	return nodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU
}

func DecodeHilbertValue(hilbertVal uint64, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU float64) (float64, float64, uint32, uint32) {
	xInt, yInt, memInt, cpuInt := decodeHilbert4D(hilbertVal)

	x := denormalize(xInt, minX, maxX, scaleMax)
	y := denormalize(yInt, minY, maxY, scaleMax)
	mem := uint32(math.Round(denormalize(memInt, minMem, maxMem, scaleMax)))
	cpu := uint32(math.Round(denormalize(cpuInt, minCPU, maxCPU, scaleMax)))

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
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d\n", node.X, node.Y, node.Memory, node.CPU)
	}

	transformedNodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU := HilbertTransform(nodes)

	fmt.Println("\nTransformed Dataset with Hilbert 1D Values:")
	for _, node := range transformedNodes {
		fmt.Printf("X: %f, Y: %f, Memory: %d, CPU: %d => Hilbert1D: %d\n",
			node.X, node.Y, node.Memory, node.CPU, node.Hilbert1D)
	}

	fmt.Println("\nDecoded values from Hilbert1D:")
	for _, node := range transformedNodes {
		x, y, mem, cpu := DecodeHilbertValue(node.Hilbert1D, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
		fmt.Printf("Hilbert1D: %d => X: %.6f, Y: %.6f, Memory: %d, CPU: %d\n",
			node.Hilbert1D, x, y, mem, cpu)
	}
}
