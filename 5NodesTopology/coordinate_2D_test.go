package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/big"
	"os"
	"os/exec"
	"strings"

	"github.com/hashicorp/serf/client"
	"github.com/hashicorp/serf/coordinate"
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
	"clab-century-serf14": "10.0.2.24",
	"clab-century-serf15": "10.0.2.25",
	"clab-century-serf16": "10.0.2.26",
	"clab-century-serf17": "10.0.2.27",
	"clab-century-serf18": "10.0.2.28",
	"clab-century-serf19": "10.0.2.29",
	"clab-century-serf20": "10.0.2.30",
	"clab-century-serf21": "10.0.2.31",
	"clab-century-serf22": "10.0.2.32",
	"clab-century-serf23": "10.0.2.33",
	"clab-century-serf24": "10.0.2.34",
	"clab-century-serf25": "10.0.2.35",
	"clab-century-serf26": "10.0.2.36",
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
	return sm.Encode(uint64(x), uint64(y)).Uint64()
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
	return rtt * 1000 // milliseconds
}

func ping(ip string) string {
	cmd := exec.Command("ping", "-c", "1", "-w", "2", ip)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("ping failed: %v", err)
	}
	return parsePingOutput(string(output))
}

func parsePingOutput(output string) string {
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "time=") {
			return strings.TrimSpace(line[strings.Index(line, "time="):])
		}
	}
	return "No RTT found"
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

	// Find min/max
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64
	for _, node := range nodes {
		minX = math.Min(minX, node.X)
		maxX = math.Max(maxX, node.X)
		minY = math.Min(minY, node.Y)
		maxY = math.Max(maxY, node.Y)
	}

	// Compute Hilbert 1D
	for i := range nodes {
		nodes[i].Hilbert1D = ComputeHilbertValue(nodes[i].X, nodes[i].Y, minX, maxX, minY, maxY)
	}

	// Extract current node
	var this Node
	for _, n := range nodes {
		if n.Name == currentNode {
			this = n
			break
		}
	}

	fmt.Printf("\nCurrent node: %s\n", currentNode)
	fmt.Println("\nComparison Table:")
	fmt.Printf("%-25s %-15s %-20s %-25s %-25s\n", "Node", "HilbertDist", "Decoded (X,Y)", "Serf RTT (ms)", "Ping RTT")

	for _, n := range nodes {
		if n.Name == this.Name {
			continue
		}

		// Hilbert distance
		hDist := math.Abs(float64(n.Hilbert1D) - float64(this.Hilbert1D))
		// Decode Hilbert back to 2D
		x, y := DecodeHilbertValue(n.Hilbert1D, minX, maxX, minY, maxY)
		// Serf RTT
		thisCoord, _ := serfClient.GetCoordinate(this.Name)
		otherCoord, _ := serfClient.GetCoordinate(n.Name)
		rtt := calculateRTT(thisCoord, otherCoord)
		// Ping
		pingRTT := ping(nodeIPs[n.Name])

		fmt.Printf("%-25s %-15.0f (%.6f, %.6f)     %-25.2f %-25s\n", n.Name, hDist, x, y, rtt, pingRTT)
	}
}
