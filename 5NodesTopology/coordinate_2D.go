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

func hilbert2D(x, y uint32) uint64 {
	sm, _ := hilbert.New(hilbertOrder, 2)
	hilbertIndex := sm.Encode(uint64(x), uint64(y))
	return hilbertIndex.Uint64()
}

func decodeHilbert2D(hilbertValue uint64) (uint32, uint32) {
	sm, _ := hilbert.New(hilbertOrder, 2)
	coords := sm.Decode(new(big.Int).SetUint64(hilbertValue))
	return uint32(coords[0]), uint32(coords[1])
}

func ComputeHilbertValue(x, y float64, minX, maxX, minY, maxY float64) uint64 {
	xInt := normalizeAndScale(x, minX, maxX)
	yInt := normalizeAndScale(y, minY, maxY)
	return hilbert2D(xInt, yInt)
}

func HilbertTransform(nodes []Node) ([]Node, float64, float64, float64, float64) {
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64

	for _, node := range nodes {
		minX = math.Min(minX, node.X)
		maxX = math.Max(maxX, node.X)
		minY = math.Min(minY, node.Y)
		maxY = math.Max(maxY, node.Y)
	}

	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].X, nodes[i].Y, minX, maxX, minY, maxY)
	}

	return nodes, minX, maxX, minY, maxY
}

func DecodeHilbertValue(hilbertVal uint64, minX, maxX, minY, maxY float64) (float64, float64) {
	xInt, yInt := decodeHilbert2D(hilbertVal)
	x := denormalize(xInt, minX, maxX)
	y := denormalize(yInt, minY, maxY)
	return x, y
}

func main() {
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
		coord, err := serfClient.GetCoordinate(member.Name)
		if err != nil || coord == nil {
			log.Printf("Warning: Coordinate unavailable for %s, skipping...", member.Name)
			continue
		}
		nodes = append(nodes, Node{
			Name: member.Name,
			X:    coord.Vec[0],
			Y:    coord.Vec[1],
		})
	}

	fmt.Println("Original Data with Coordinates:")
	for _, node := range nodes {
		fmt.Printf("%s -> X: %.6f, Y: %.6f\n", node.Name, node.X, node.Y)
	}

	nodes, minX, maxX, minY, maxY := HilbertTransform(nodes)

	// ✅ Print all calculated 1D Hilbert values
	fmt.Println("\nCalculated Hilbert 1D Values:")
	for _, node := range nodes {
		fmt.Printf("%s => Hilbert1D: %d\n", node.Name, node.Hilbert1D)
	}

	// Ask the user to select a node
	var selectedName string
	fmt.Print("\nEnter the name of the node to find closest Hilbert neighbors: ")
	fmt.Scanln(&selectedName)

	// Find the selected node
	var selectedNode *Node
	for i := range nodes {
		if nodes[i].Name == selectedName {
			selectedNode = &nodes[i]
			break
		}
	}

	if selectedNode == nil {
		log.Fatalf("Node %s not found.", selectedName)
	}

	type NodeDistance struct {
		Node     Node
		Distance uint64
	}
	var distances []NodeDistance
	for _, node := range nodes {
		if node.Name == selectedNode.Name {
			continue
		}
		distance := uint64(math.Abs(float64(node.Hilbert1D) - float64(selectedNode.Hilbert1D)))

		distances = append(distances, NodeDistance{Node: node, Distance: distance})
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})

	fmt.Printf("\nNodes closest to '%s' by Hilbert 1D value:\n", selectedNode.Name)
	for _, nd := range distances {
		x, y := DecodeHilbertValue(nd.Node.Hilbert1D, minX, maxX, minY, maxY)
		fmt.Printf("%s => Hilbert1D: %d, Distance: %d, Decoded (X,Y): (%.6f, %.6f)\n",
			nd.Node.Name, nd.Node.Hilbert1D, nd.Distance, x, y)
	}
}
