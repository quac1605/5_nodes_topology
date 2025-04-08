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
	Memory    uint32
	CPU       uint32
	Hilbert1D uint64
}

// Static memory and CPU map per node name
var resourceMap = map[string]struct {
	Memory uint32
	CPU    uint32
}{
	"clab-century-serf1": {Memory: 128, CPU: 12},
	"clab-century-serf2": {Memory: 256, CPU: 20},
	"clab-century-serf3": {Memory: 512, CPU: 8},
	"clab-century-serf4": {Memory: 8, CPU: 8},
	"clab-century-serf5": {Memory: 1024, CPU: 16},
}

func normalizeAndScale(value, min, max float64) uint32 {
	if max == min {
		return 0
	}
	normalized := (value - min) / (max - min)
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
	// Connect to Serf client
	serfClient, err := client.NewRPCClient("127.0.0.1:7373")
	if err != nil {
		log.Fatalf("Failed to connect to Serf agent: %v", err)
	}
	defer serfClient.Close()

	serfMembers, err := serfClient.Members()
	if err != nil {
		log.Fatalf("Failed to get members: %v", err)
	}

	var nodes []Node
	for _, member := range serfMembers {
		if res, ok := resourceMap[member.Name]; ok {
			coord, err := serfClient.GetCoordinate(member.Name)
			if err != nil || coord == nil {
				log.Printf("Warning: Coordinate unavailable for %s, skipping...", member.Name)
				continue
			}
			nodes = append(nodes, Node{
				Name:   member.Name,
				X:      coord.Vec[0],
				Y:      coord.Vec[1],
				Memory: res.Memory,
				CPU:    res.CPU,
			})
		}
	}

	fmt.Println("Original Data with Coordinates:")
	for _, node := range nodes {
		fmt.Printf("%s -> X: %.6f, Y: %.6f, Memory: %d, CPU: %d\n", node.Name, node.X, node.Y, node.Memory, node.CPU)
	}

	transformedNodes, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU := HilbertTransform(nodes)

	fmt.Println("\nSorted Nodes by Hilbert 1D Value:")
	for _, node := range transformedNodes {
		fmt.Printf("%s => Hilbert1D: %d\n", node.Name, node.Hilbert1D)
	}

	fmt.Println("\nDecoded from Hilbert 1D:")
	for _, node := range transformedNodes {
		x, y, mem, cpu := DecodeHilbertValue(node.Hilbert1D, minX, maxX, minY, maxY, minMem, maxMem, minCPU, maxCPU)
		fmt.Printf("%s => X: %.6f, Y: %.6f, Mem: %.2f, CPU: %.2f\n", node.Name, x, y, mem, cpu)
	}
}
