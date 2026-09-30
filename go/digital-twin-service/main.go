// digital-twin-service maintains the authoritative, in-memory replica of the
// hybrid 5G/6G + LEO humanitarian network. It is the single source of truth for
// node telemetry, model-derived risk scores and the current topology, and it is
// the only component the orchestrator has to talk to when computing paths.
//
// Endpoints
//
//	GET  /healthz                 liveness
//	GET  /readyz                  readiness
//	GET  /metrics                 Prometheus exposition
//	GET  /api/v1/nodes            all nodes
//	GET  /api/v1/nodes/{id}       one node
//	POST /api/v1/nodes/{id}/telemetry  ingest a telemetry sample
//	POST /api/v1/risk             bulk risk update from predictor-service
//	GET  /api/v1/topology         nodes + links
//	GET  /api/v1/state            full snapshot (nodes, links, risk, load)
//	POST /api/v1/reset            clear runtime state (test helper)
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	observedFailLoss  = 0.50
	observedFailAvail = 0.08
	detectionDelay    = 3
)

// Telemetry is one observation of a network element.
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
	ObservedAt     int64   `json:"observed_at"`
}

// Node is the twin of a physical network element.
type Node struct {
	ID                 string    `json:"id"`
	Type               string    `json:"type"`
	Layer              string    `json:"layer"`
	CapacityMbps       float64   `json:"capacity_mbps"`
	LoadMbps           float64   `json:"load_mbps"`
	FailureProbability float64   `json:"failure_probability"`
	Status             string    `json:"status"`
	Telemetry          Telemetry `json:"telemetry"`
	consecutiveBad     int
	UpdatedAt          time.Time `json:"updated_at"`
}

// Link is an undirected edge of the topology.
type Link struct {
	Source       string  `json:"source"`
	Target       string  `json:"target"`
	PropLatency  float64 `json:"prop_latency_ms"`
	CapacityMbps float64 `json:"capacity_mbps"`
	LoadMbps     float64 `json:"load_mbps"`
}

// Twin is the concurrency-safe store.
type Twin struct {
	mu      sync.RWMutex
	nodes   map[string]*Node
	links   []*Link
	ingests uint64
	risks   uint64
	started time.Time
}

func NewTwin() *Twin {
	t := &Twin{nodes: map[string]*Node{}, started: time.Now()}
	t.seed()
	return t
}

type nodeSeed struct {
	id, typ, layer string
	capacity       float64
}

func (t *Twin) seed() {
	seeds := []nodeSeed{
		{"hq_1", "HQ", "core", 2000},
		{"edge_1", "EDGE", "edge", 900}, {"edge_2", "EDGE", "edge", 900},
		{"leo_1", "LEO", "non_terrestrial", 320}, {"leo_2", "LEO", "non_terrestrial", 320},
		{"leo_3", "LEO", "non_terrestrial", 320},
		{"bs6g_1", "6G", "terrestrial_access", 1400}, {"bs6g_2", "6G", "terrestrial_access", 1400},
		{"bs5g_1", "5G", "terrestrial_access", 600}, {"bs5g_2", "5G", "terrestrial_access", 600},
		{"bs5g_3", "5G", "terrestrial_access", 600},
		{"bs4g_1", "4G", "terrestrial_access", 120}, {"bs4g_2", "4G", "terrestrial_access", 120},
		{"uav_1", "UAV", "aerial", 180}, {"uav_2", "UAV", "aerial", 180},
		{"term_1", "MOBILE", "user", 200}, {"term_2", "MOBILE", "user", 200},
		{"term_3", "MOBILE", "user", 200}, {"term_4", "MOBILE", "user", 200},
	}
	for _, s := range seeds {
		t.nodes[s.id] = &Node{
			ID: s.id, Type: s.typ, Layer: s.layer, CapacityMbps: s.capacity,
			Status: "healthy", UpdatedAt: time.Now(),
		}
	}
	edges := []Link{
		{"term_1", "bs6g_1", 2.0, 480, 0}, {"term_1", "bs5g_1", 3.0, 260, 0},
		{"term_1", "bs4g_1", 5.0, 70, 0}, {"term_2", "bs5g_1", 3.0, 260, 0},
		{"term_2", "bs5g_2", 3.5, 240, 0}, {"term_2", "uav_1", 6.0, 120, 0},
		{"term_3", "bs5g_3", 4.0, 220, 0}, {"term_3", "bs4g_2", 5.5, 70, 0},
		{"term_3", "uav_2", 6.5, 120, 0}, {"term_4", "bs6g_2", 2.5, 460, 0},
		{"term_4", "bs5g_3", 4.0, 220, 0},
		{"term_1", "leo_1", 24.0, 90, 0}, {"term_2", "leo_2", 26.0, 90, 0},
		{"term_3", "leo_2", 25.0, 90, 0}, {"term_4", "leo_3", 27.0, 90, 0},
		{"uav_1", "bs5g_2", 4.0, 150, 0}, {"uav_1", "edge_1", 8.0, 150, 0},
		{"uav_1", "leo_1", 20.0, 80, 0}, {"uav_2", "bs5g_3", 4.5, 150, 0},
		{"uav_2", "edge_2", 9.0, 150, 0}, {"uav_2", "leo_3", 21.0, 80, 0},
		{"bs6g_1", "edge_1", 3.0, 1200, 0}, {"bs6g_2", "edge_2", 3.0, 1200, 0},
		{"bs5g_1", "edge_1", 6.0, 520, 0}, {"bs5g_2", "edge_1", 6.5, 520, 0},
		{"bs5g_3", "edge_2", 6.0, 520, 0}, {"bs4g_1", "edge_1", 12.0, 110, 0},
		{"bs4g_2", "edge_2", 12.0, 110, 0}, {"edge_1", "edge_2", 5.0, 900, 0},
		{"edge_1", "hq_1", 9.0, 1400, 0}, {"edge_2", "hq_1", 10.0, 1400, 0},
		{"leo_1", "hq_1", 22.0, 260, 0}, {"leo_2", "hq_1", 23.0, 260, 0},
		{"leo_3", "hq_1", 24.0, 260, 0}, {"leo_1", "leo_2", 12.0, 300, 0},
		{"leo_2", "leo_3", 12.0, 300, 0},
	}
	for i := range edges {
		e := edges[i]
		t.links = append(t.links, &e)
	}
}

func (t *Twin) IngestTelemetry(id string, tel Telemetry) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	n, ok := t.nodes[id]
	if !ok {
		return fmt.Errorf("unknown node %q", id)
	}
	tel.ObservedAt = time.Now().Unix()
	n.Telemetry = tel
	n.LoadMbps = tel.Utilisation * n.CapacityMbps
	if tel.PacketLoss >= observedFailLoss && tel.Availability <= observedFailAvail {
		n.consecutiveBad++
	} else {
		n.consecutiveBad = 0
	}
	switch {
	case n.consecutiveBad > detectionDelay:
		n.Status = "failed"
	case n.FailureProbability >= 0.5 || tel.PacketLoss > 0.10:
		n.Status = "degraded"
	default:
		n.Status = "healthy"
	}
	n.UpdatedAt = time.Now()
	t.ingests++
	return nil
}

type riskUpdate struct {
	Risks map[string]float64 `json:"risks"`
}

func (t *Twin) UpdateRisk(u riskUpdate) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	applied := 0
	for id, r := range u.Risks {
		if n, ok := t.nodes[id]; ok {
			n.FailureProbability = r
			applied++
		}
	}
	t.risks++
	return applied
}

func (t *Twin) Nodes() []*Node {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]*Node, 0, len(t.nodes))
	for _, n := range t.nodes {
		cp := *n
		out = append(out, &cp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (t *Twin) Node(id string) (*Node, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	n, ok := t.nodes[id]
	if !ok {
		return nil, false
	}
	cp := *n
	return &cp, true
}

type topology struct {
	Nodes []*Node `json:"nodes"`
	Links []*Link `json:"links"`
}

func (t *Twin) Topology() topology {
	t.mu.RLock()
	defer t.mu.RUnlock()
	links := make([]*Link, 0, len(t.links))
	for _, l := range t.links {
		cp := *l
		links = append(links, &cp)
	}
	return topology{Nodes: t.Nodes(), Links: links}
}

// ------------------------------------------------------------------ HTTP layer

type server struct {
	twin *Twin
	log  *slog.Logger
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"nodes": len(s.twin.Nodes())})
	})
	mux.HandleFunc("GET /api/v1/nodes", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.twin.Nodes())
	})
	mux.HandleFunc("GET /api/v1/nodes/{id}", func(w http.ResponseWriter, r *http.Request) {
		n, ok := s.twin.Node(r.PathValue("id"))
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
			return
		}
		writeJSON(w, http.StatusOK, n)
	})
	mux.HandleFunc("POST /api/v1/nodes/{id}/telemetry", func(w http.ResponseWriter, r *http.Request) {
		var tel Telemetry
		if err := json.NewDecoder(r.Body).Decode(&tel); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if err := s.twin.IngestTelemetry(r.PathValue("id"), tel); err != nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "accepted"})
	})
	mux.HandleFunc("POST /api/v1/risk", func(w http.ResponseWriter, r *http.Request) {
		var u riskUpdate
		if err := json.NewDecoder(r.Body).Decode(&u); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]int{"applied": s.twin.UpdateRisk(u)})
	})
	mux.HandleFunc("GET /api/v1/topology", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.twin.Topology())
	})
	mux.HandleFunc("GET /api/v1/state", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.twin.Topology())
	})
	mux.HandleFunc("GET /metrics", s.metrics)
	return logging(s.log, mux)
}

func (s *server) metrics(w http.ResponseWriter, r *http.Request) {
	var b strings.Builder
	s.twin.mu.RLock()
	defer s.twin.mu.RUnlock()
	b.WriteString("# HELP dt_node_failure_probability Model-derived failure probability\n")
	b.WriteString("# TYPE dt_node_failure_probability gauge\n")
	for id, n := range s.twin.nodes {
		fmt.Fprintf(&b, "dt_node_failure_probability{node=%q,type=%q} %f\n", id, n.Type, n.FailureProbability)
	}
	b.WriteString("# HELP dt_node_latency_ms Observed node latency\n# TYPE dt_node_latency_ms gauge\n")
	for id, n := range s.twin.nodes {
		fmt.Fprintf(&b, "dt_node_latency_ms{node=%q} %f\n", id, n.Telemetry.LatencyMs)
	}
	b.WriteString("# HELP dt_telemetry_ingests_total Telemetry samples ingested\n# TYPE dt_telemetry_ingests_total counter\n")
	fmt.Fprintf(&b, "dt_telemetry_ingests_total %d\n", s.twin.ingests)
	fmt.Fprintf(&b, "dt_uptime_seconds %f\n", time.Since(s.twin.started).Seconds())
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(b.String()))
}

func logging(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Info("request", "method", r.Method, "path", r.URL.Path,
			"duration_ms", float64(time.Since(start).Microseconds())/1000.0)
	})
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	s := &server{twin: NewTwin(), log: log}
	log.Info("digital-twin-service starting", "addr", addr)
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	if err := srv.ListenAndServe(); err != nil {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
