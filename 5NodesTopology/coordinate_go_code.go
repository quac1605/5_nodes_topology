package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	"github.com/google/hilbert"
	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
)

const hilbertOrder = 16
const scaleMax = 255 // Maximum value for scaling (255 for 256x256 Hilbert space)

type Node struct {
	Name        string
	X           float64
	Y           float64
	Hilbert1D   uint64
	RTT         float64
	PingRTT     float64
	PingResult  string
	HilbertDist float64
}

var nodeIPs = map[string]string{
	"clab-century-serf1":  "10.0.1.11",
	"clab-century-serf2":  "10.0.1.12",
	"clab-century-serf3":  "10.0.1.13",
	"clab-century-serf4":  "10.0.1.14",
	"clab-century-serf5":  "10.0.1.15",
	"clab-century-serf6":  "10.0.1.16",
	"clab-century-serf7":  "10.0.1.17",
	"clab-century-serf8":  "10.0.1.18",
	"clab-century-serf9":  "10.0.1.19",
	"clab-century-serf10": "10.0.1.20",
	"clab-century-serf11": "10.0.1.21",
	"clab-century-serf12": "10.0.1.22",
	"clab-century-serf13": "10.0.1.23",
	"clab-century-serf14": "10.0.1.24",
	"clab-century-serf15": "10.0.1.25",
	"clab-century-serf16": "10.0.1.26",
	"clab-century-serf17": "10.0.1.27",
	"clab-century-serf18": "10.0.1.28",
	"clab-century-serf19": "10.0.1.29",
	"clab-century-serf20": "10.0.1.30",
}

func normalizeAndScale(value, min, max float64) uint32 {
	if max == min {
		return 0
	}
	normalized := (value - min) / (max - min)
	normalized = math.Max(0, math.Min(1, normalized))
	// Scale the value to the 0-255 range for the Hilbert curve
	return uint32(math.Round(normalized * 255)) // Maximum value is 255 for a 256x256 grid
}

func denormalize(value uint32, min, max float64) float64 {
	if scaleMax == 0 {
		return min
	}
	normalized := float64(value) / float64(scaleMax)
	return min + normalized*(max-min)
}

func hilbert2D(x, y uint32) uint64 {
	hilbertSpace, _ := hilbert.NewHilbert(scaleMax + 1) // Initialize Hilbert with size 256
	t, err := hilbertSpace.MapInverse(int(x), int(y))   // Correctly handle the two return values
	if err != nil {
		log.Fatalf("Error in Hilbert MapInverse: %v", err)
	}
	return uint64(t)
}

func decodeHilbert2D(hilbertValue uint64) (uint32, uint32) {
	hilbertSpace, _ := hilbert.NewHilbert(scaleMax + 1)
	x, y, err := hilbertSpace.Map(int(hilbertValue)) // Correctly handle the two return values
	if err != nil {
		log.Fatalf("Error in Hilbert Map: %v", err)
	}
	return uint32(x), uint32(y)
}

func ComputeHilbertValue(x, y float64, minX, maxX, minY, maxY float64) uint64 {
	xInt := normalizeAndScale(x, minX, maxX)
	yInt := normalizeAndScale(y, minY, maxY)
	return hilbert2D(xInt, yInt)
}

func DecodeHilbertValue(hilbertVal uint64, minX, maxX, minY, maxY float64) (float64, float64) {
	xInt, yInt := decodeHilbert2D(hilbertVal)
	return denormalize(xInt, minX, maxX), denormalize(yInt, minY, maxY)
}

func calculateRTT(a, b *coordinate.Coordinate) float64 {
	if len(a.Vec) != len(b.Vec) {
		panic("coordinate dimensions do not match")
	}
	sumsq := 0.0
	for i := 0; i < len(a.Vec); i++ {
		diff := a.Vec[i] - b.Vec[i]
		sumsq += diff * diff
	}
	rtt := math.Sqrt(sumsq) + a.Height + b.Height
	adjusted := rtt + a.Adjustment + b.Adjustment
	if adjusted > 0.0 {
		rtt = adjusted
	}
	return rtt * 1000 // ms
}

func ping(ip string) (string, float64) {
	cmd := exec.Command("ping", "-c", "1", "-w", "2", ip)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("ping failed: %v", err), -1
	}
	return parsePingOutput(string(output))
}

func parsePingOutput(output string) (string, float64) {
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "time=") {
			timeStr := line[strings.Index(line, "time=")+5:]
			timeStr = strings.TrimSpace(timeStr)
			timeStr = strings.Split(timeStr, " ")[0]
			if ms, err := strconv.ParseFloat(timeStr, 64); err == nil {
				return "time=" + timeStr + " ms", ms
			}
		}
	}
	return "No RTT found", -1
}

func main() {
	serfClient, err := client.NewRPCClient("127.0.0.1:7373")
	if err != nil {
		log.Fatalf("Failed to connect to Serf: %v", err)
	}
	defer serfClient.Close()

	hostname, _ := os.Hostname()
	members, err := serfClient.Members()
	if err != nil {
		log.Fatalf("Failed to get members: %v", err)
	}

	var currentNode string
	var nodes []Node

	for _, member := range members {
		if strings.HasSuffix(member.Name, hostname) {
			currentNode = member.Name
		}
		coord, err := serfClient.GetCoordinate(member.Name)
		if err != nil || coord == nil {
			log.Printf("Skipping %s, no coordinates", member.Name)
			continue
		}
		nodes = append(nodes, Node{
			Name: member.Name,
			X:    coord.Vec[0],
			Y:    coord.Vec[1],
		})
	}

	if currentNode == "" {
		log.Fatalf("Could not determine the current node")
	}

	thisCoord, _ := serfClient.GetCoordinate(currentNode)

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

	var thisNode Node
	for _, n := range nodes {
		if n.Name == currentNode {
			thisNode = n
			break
		}
	}

	var filtered []Node
	for _, node := range nodes {
		if node.Name == thisNode.Name {
			continue
		}
		coord, _ := serfClient.GetCoordinate(node.Name)
		node.RTT = calculateRTT(thisCoord, coord)
		node.HilbertDist = math.Abs(float64(node.Hilbert1D) - float64(thisNode.Hilbert1D))
		ip := nodeIPs[node.Name]
		node.PingResult, node.PingRTT = ping(ip)
		filtered = append(filtered, node)
	}

	fmt.Printf("Current Node: %s\n\n", currentNode)

	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].RTT < filtered[j].RTT
	})
	fmt.Println("1. Distance through Round Trip Time (ms):")
	for _, n := range filtered {
		fmt.Printf("   %-25s => %.2f ms\n", n.Name, n.RTT)
	}

	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].PingRTT < filtered[j].PingRTT
	})
	fmt.Println("\n2. Distance through Ping:")
	for _, n := range filtered {
		fmt.Printf("   %-25s => %s\n", n.Name, n.PingResult)
	}

	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].HilbertDist < filtered[j].HilbertDist
	})
	fmt.Println("\n3. Distance with Hilbert 1D Transform:")
	for _, n := range filtered {
		decodedX, decodedY := DecodeHilbertValue(n.Hilbert1D, minX, maxX, minY, maxY)
		fmt.Printf("   %-25s => Hilbert1D: %-10d HilbertDist: %-10.0f Decoded(X,Y): (%.6f, %.6f) Original(X,Y): (%.6f, %.6f)\n",
			n.Name, n.Hilbert1D, n.HilbertDist, decodedX, decodedY, n.X, n.Y)
	}
}