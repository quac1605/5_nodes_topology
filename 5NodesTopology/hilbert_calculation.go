package main

import (
	"fmt"
	"math"
)

// Function to compute the Hilbert index from coordinates in a 5D space
func hilbertIndex(coordinates []int, p int) int {
	nDims := len(coordinates)
	hIndex := 0

	// Combine the coordinates to generate a Hilbert index using bitwise operations
	for i := 0; i < nDims; i++ {
		hIndex |= coordinates[i] << (i * p)
	}
	return hIndex
}

// Normalize and shift data to handle negative values
func normalizeAndShiftData(data [][]int, p int) [][]int {
	// Find the minimum value in each dimension
	minValues := make([]int, len(data[0]))

	// Initialize min values to a very large value
	for i := 0; i < len(minValues); i++ {
		minValues[i] = int(math.MaxInt32)
	}

	// Find the minimum for each dimension
	for _, point := range data {
		for i, val := range point {
			if val < minValues[i] {
				minValues[i] = val
			}
		}
	}

	// Normalize data to the range [0, 2^p - 1]
	maxVal := int(math.Pow(2, float64(p)) - 1)
	normalizedData := make([][]int, len(data))

	for i, point := range data {
		normalizedData[i] = make([]int, len(point))
		for j, val := range point {
			// Shift the value to make all values non-negative
			shiftedVal := val - minValues[j]

			// Scale to fit the range [0, 2^p - 1]
			normalizedData[i][j] = int(float64(shiftedVal) / float64(maxVal) * float64(maxVal))
		}
	}

	return normalizedData
}

func main() {
	// Example 5D data points with negative values
	data := [][]int{
		{12, -34, 56, 78, -90},
		{-9, 45, -67, 89, 10},
	}

	// Hilbert curve order (e.g., p = 5 for 32x32 grid)
	p := 5

	// Normalize and shift data to fit the 2^p grid and handle negative values
	normalizedData := normalizeAndShiftData(data, p)

	// Map each point to a 1D Hilbert index
	for _, point := range normalizedData {
		hIndex := hilbertIndex(point, p)
		fmt.Printf("Hilbert Index for point %v: %d\n", point, hIndex)
	}
}
