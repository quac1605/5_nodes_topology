package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Structure to hold the resource information of each container
type ContainerStats struct {
	Name         string
	RemainingCPU string
	RemainingMem string
}

// Convert memory strings to float for calculation (e.g., "1.397GiB" -> 1.397)
func convertMemoryToFloat(memoryStr string) (float64, string, error) {
	var factor float64
	unit := memoryStr[len(memoryStr)-3:]

	// Convert the memory to the base unit (MB or GiB)
	switch unit {
	case "GiB":
		factor = 1024 // GiB to MiB conversion factor
	case "MiB":
		factor = 1 // No conversion needed
	default:
		return 0, "", fmt.Errorf("unsupported memory unit: %s", unit)
	}

	// Remove the unit and convert the number
	numStr := memoryStr[:len(memoryStr)-3]
	num, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, "", fmt.Errorf("error converting memory to float: %v", err)
	}

	// Return the memory in MiB
	return num * factor, unit, nil
}

// Collect resource data (CPU and memory usage)
func collectResources() ([]ContainerStats, error) {
	// Execute the "docker stats --no-stream" command
	cmd := exec.Command("docker", "stats", "--no-stream")
	output, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("error executing command: %v", err)
	}
	defer output.Close()

	// Run the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("error starting the command: %v", err)
	}

	// Prepare to parse the output
	scanner := bufio.NewScanner(output)
	var containerStats []ContainerStats

	// Skip the first line (header)
	scanner.Scan()

	// Process each line of output
	for scanner.Scan() {
		line := scanner.Text()
		// Skip empty lines
		if len(line) == 0 {
			continue
		}

		// Split the line by whitespace
		parts := strings.Fields(line)

		// Ensure that we have enough parts (container name, CPU%, MEM usage)
		if len(parts) < 6 {
			continue
		}

		// Extract the container name, CPU %, and Memory usage
		name := parts[1]
		cpuUsageStr := parts[2]
		memoryUsageStr := parts[3] + " " + parts[4] + " " + parts[5] // MEM USAGE / LIMIT

		// Convert the CPU usage string to a float
		cpuUsage, err := strconv.ParseFloat(cpuUsageStr[:len(cpuUsageStr)-1], 64) // Removing '%' sign
		if err != nil {
			return nil, fmt.Errorf("error parsing CPU usage: %v", err)
		}

		// Calculate the remaining CPU usage
		remainingCPU := 100.0 - cpuUsage

		// Extract total and used memory and calculate remaining memory
		partsMemory := strings.Fields(memoryUsageStr)
		totalMemory := partsMemory[2]
		usedMemory := partsMemory[0]

		// Convert the memory usage and total memory to float
		usedMem, unit, err := convertMemoryToFloat(usedMemory)
		if err != nil {
			return nil, err
		}
		totalMem, _, err := convertMemoryToFloat(totalMemory)
		if err != nil {
			return nil, err
		}

		// Calculate remaining memory
		remainingMem := totalMem - usedMem

		// Format remaining memory
		remainingMemStr := fmt.Sprintf("%.2f %s", remainingMem, unit)

		// Append the stats to the slice
		containerStats = append(containerStats, ContainerStats{
			Name:         name,
			RemainingCPU: fmt.Sprintf("%.2f%%", remainingCPU),
			RemainingMem: remainingMemStr,
		})
	}

	// Handle any errors while scanning
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading output: %v", err)
	}

	return containerStats, nil
}

// Copy the stats file to the container, if the container exists
func copyFileToContainer(filePath string, containerName string) error {
	// Check if the container exists
	cmd := exec.Command("docker", "ps", "-q", "-f", fmt.Sprintf("name=%s", containerName))
	output, err := cmd.CombinedOutput()
	if err != nil || len(output) == 0 {
		return fmt.Errorf("container %s does not exist", containerName)
	}

	// Copy the file to the container
	copyCmd := exec.Command("docker", "cp", filePath, fmt.Sprintf("%s:/opt/container_stats.txt", containerName))
	err = copyCmd.Run()
	if err != nil {
		return fmt.Errorf("error copying file to container: %v", err)
	}

	return nil
}

func main() {
	// Run the command and save data every 10 seconds
	for {
		// Collect resource data
		stats, err := collectResources()
		if err != nil {
			fmt.Println("Error collecting resource data:", err)
			return
		}

		// Get the current working directory
		currentDir, err := os.Getwd()
		if err != nil {
			fmt.Println("Error getting current directory:", err)
			return
		}

		// Define the file path relative to the current directory
		filePath := filepath.Join(currentDir, "container_stats.txt")

		// Create the file in the current directory
		file, err := os.Create(filePath)
		if err != nil {
			fmt.Println("Error creating file:", err)
			return
		}
		defer file.Close()

		// Write the stats to the file
		for _, stat := range stats {
			_, err := file.WriteString(fmt.Sprintf("Name: %s, Remaining CPU: %s, Remaining Memory: %s\n", stat.Name, stat.RemainingCPU, stat.RemainingMem))
			if err != nil {
				fmt.Println("Error writing to file:", err)
				return
			}
		}

		// Print the data to the console
		fmt.Println("Remaining resource usage data collected:")
		for _, stat := range stats {
			fmt.Printf("Name: %s, Remaining CPU: %s, Remaining Memory: %s\n", stat.Name, stat.RemainingCPU, stat.RemainingMem)
		}

		// Copy the stats file to the container "clab-century-serf1"
		err = copyFileToContainer(filePath, "clab-century-serf1")
		if err != nil {
			fmt.Println("Error copying file to container:", err)
		} else {
			fmt.Println("File successfully copied to container.")
		}

		// Wait for 10 seconds before collecting the data again
		fmt.Println("Waiting for 10 seconds...\n")
		time.Sleep(10 * time.Second)
	}
}
