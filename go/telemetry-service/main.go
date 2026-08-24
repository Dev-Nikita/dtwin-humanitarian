// telemetry-service is the collection agent of the framework. In the reference
// deployment it runs as a DaemonSet next to the network elements, samples their
// KPIs, and forwards them to digital-twin-service. For the reproducible
// experiments it can also replay a recorded telemetry trace (CSV) so that the
// Go control plane can be exercised against exactly the same data that the
// Python evaluation uses.
//
// Endpoints
//
//	GET  /healthz
//	GET  /metrics
//	POST /api/v1/replay      start replaying a CSV trace
//	POST /api/v1/sample      push one sample through the agent
//	GET  /api/v1/status      replay progress
package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

type Telemetry struct {
	LatencyMs      float64 `json:"latency_ms"`
	JitterMs       float64 `json:"jitter_ms"`
	PacketLoss     float64 `json:"packet_loss"`
	ThroughputMbps float64 `json:"throughput_mbps"`
	CPUUsage       float64 `json:"cpu_usage"`
	MemoryUsage    float64 `json:"memory_usage"`
	SignalStrength float64 `json:"signal_strength"`
	Availability   float64 `json:"availability"`
	Utilisation    float64 `json:"utilisation"`
	QueueDepth     float64 `json:"queue_depth"`
	RetxRate       float64 `json:"retx_rate"`
	EnergyHeadroom float64 `json:"energy_headroom"`
}

type Sample struct {
	NodeID    string    `json:"node_id"`
	Telemetry Telemetry `json:"telemetry"`
}

type agent struct {
	mu        sync.RWMutex
	twinURL   string
	client    *http.Client
	log       *slog.Logger
	sent      int
	errors    int
	replaying bool
	progress  int
	total     int
}

func (a *agent) forward(s Sample) error {
	body, _ := json.Marshal(s.Telemetry)
	url := fmt.Sprintf("%s/api/v1/nodes/%s/telemetry", a.twinURL, s.NodeID)
	resp, err := a.client.Post(url, "application/json", bytes.NewReader(body))
	a.mu.Lock()
	defer a.mu.Unlock()
	if err != nil {
		a.errors++
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		a.errors++
		return fmt.Errorf("twin returned %s", resp.Status)
	}
	a.sent++
	return nil
}

// replay streams a CSV produced by the Python simulator into the twin.
func (a *agent) replay(path string, slotMillis int) error {
	fh, err := os.Open(path)
	if err != nil {
		return err
	}
	defer fh.Close()
	reader := csv.NewReader(bufio.NewReader(fh))
	header, err := reader.Read()
	if err != nil {
		return err
	}
	idx := map[string]int{}
	for i, h := range header {
		idx[h] = i
	}
	need := []string{"node_id", "latency_ms", "jitter_ms", "packet_loss", "throughput_mbps",
		"cpu_usage", "memory_usage", "signal_strength", "availability", "utilisation",
		"queue_depth", "retx_rate", "energy_headroom"}
	for _, k := range need {
		if _, ok := idx[k]; !ok {
			return fmt.Errorf("column %q missing in trace", k)
		}
	}
	records, err := reader.ReadAll()
	if err != nil {
		return err
	}
	a.mu.Lock()
	a.replaying, a.total, a.progress = true, len(records), 0
	a.mu.Unlock()

	num := func(rec []string, key string) float64 {
		v, _ := strconv.ParseFloat(rec[idx[key]], 64)
		return v
	}
	for _, rec := range records {
		s := Sample{NodeID: rec[idx["node_id"]], Telemetry: Telemetry{
			LatencyMs: num(rec, "latency_ms"), JitterMs: num(rec, "jitter_ms"),
			PacketLoss: num(rec, "packet_loss"), ThroughputMbps: num(rec, "throughput_mbps"),
			CPUUsage: num(rec, "cpu_usage"), MemoryUsage: num(rec, "memory_usage"),
			SignalStrength: num(rec, "signal_strength"), Availability: num(rec, "availability"),
			Utilisation: num(rec, "utilisation"), QueueDepth: num(rec, "queue_depth"),
			RetxRate: num(rec, "retx_rate"), EnergyHeadroom: num(rec, "energy_headroom"),
		}}
		if err := a.forward(s); err != nil {
			a.log.Warn("forward failed", "node", s.NodeID, "err", err)
		}
		a.mu.Lock()
		a.progress++
		a.mu.Unlock()
		if slotMillis > 0 {
			time.Sleep(time.Duration(slotMillis) * time.Millisecond)
		}
	}
	a.mu.Lock()
	a.replaying = false
	a.mu.Unlock()
	return nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	twin := os.Getenv("TWIN_URL")
	if twin == "" {
		twin = "http://digital-twin-service:8080"
	}
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8083"
	}
	a := &agent{twinURL: twin, client: &http.Client{Timeout: 3 * time.Second}, log: log}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /api/v1/sample", func(w http.ResponseWriter, r *http.Request) {
		var s Sample
		if err := json.NewDecoder(r.Body).Decode(&s); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if err := a.forward(s); err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "forwarded"})
	})
	mux.HandleFunc("POST /api/v1/replay", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Path       string `json:"path"`
			SlotMillis int    `json:"slot_millis"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		go func() {
			if err := a.replay(body.Path, body.SlotMillis); err != nil {
				log.Error("replay failed", "err", err)
			}
		}()
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "replay started"})
	})
	mux.HandleFunc("GET /api/v1/status", func(w http.ResponseWriter, r *http.Request) {
		a.mu.RLock()
		defer a.mu.RUnlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"replaying": a.replaying, "progress": a.progress,
			"total": a.total, "sent": a.sent, "errors": a.errors,
		})
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		a.mu.RLock()
		defer a.mu.RUnlock()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "telemetry_samples_sent_total %d\ntelemetry_errors_total %d\n", a.sent, a.errors)
	})

	log.Info("telemetry-service starting", "addr", addr, "twin", twin)
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	if err := srv.ListenAndServe(); err != nil {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
