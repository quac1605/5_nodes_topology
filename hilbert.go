package main

import (
	"fmt"
	"math"
)

type Hilbert struct {
	BitsPerDimension int
	NumDimensions    int
}

func (h *Hilbert) distanceFromCoordinates(coords []int) int {
	order := int(math.Pow(2, float64(h.BitsPerDimension)))
	hilbertIndex := 0
	for i, coord := range coords {
		hilbertIndex += coord * int(math.Pow(float64(order), float64(h.NumDimensions-i-1)))
	}
	return hilbertIndex
}

func normalize(value float64, max int) int {
	return int(value * float64(max))
}

func main() {
	h := Hilbert{BitsPerDimension: 16, NumDimensions: 4}

	vivaldiCoords := []float64{0.1, 0.5}
	cpuAvailability := 0.7
	memAvailability := 0.6

	coords := []int{
		normalize(vivaldiCoords[0], 65535),
		normalize(vivaldiCoords[1], 65535),
		normalize(cpuAvailability, 65535),
		normalize(memAvailability, 65535),
	}

	hilbertIndex := h.distanceFromCoordinates(coords)
	fmt.Printf("Hilbert Index: %d\n", hilbertIndex)
}
