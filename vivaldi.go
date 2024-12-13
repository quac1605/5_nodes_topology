package main

import (
	"fmt"
	"math"
)

type VivaldiNode struct {
	Position [2]float64
	Weight   float64
}

func (node *VivaldiNode) Update(latency float64, remotePosition [2]float64, remoteWeight float64) {
	c := 0.25 // Convergence factor
	error := latency - node.distanceTo(remotePosition)
	adjustment := c * error / (node.Weight + remoteWeight)

	node.Position[0] += adjustment * (remotePosition[0] - node.Position[0])
	node.Position[1] += adjustment * (remotePosition[1] - node.Position[1])
}

func (node *VivaldiNode) distanceTo(remotePosition [2]float64) float64 {
	return math.Sqrt(math.Pow(remotePosition[0]-node.Position[0], 2) + math.Pow(remotePosition[1]-node.Position[1], 2))
}

func main() {
	node := VivaldiNode{Position: [2]float64{0.0, 0.0}, Weight: 1.0}
	peerPosition := [2]float64{0.5, 0.5}
	peerWeight := 1.0
	latency := 20.0

	node.Update(latency, peerPosition, peerWeight)
	fmt.Printf("Updated Position: %+v\n", node.Position)
}
