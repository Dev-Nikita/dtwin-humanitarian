// orchestrator-service implements the risk-aware service orchestration layer.
// It pulls the topology and the current risk vector from digital-twin-service,
// computes SLA-constrained least-cost paths for humanitarian service requests
// and performs pre-emptive migration of flows whose bottleneck risk exceeds the
// service-differentiated budget tau_c.
//
// Endpoints
//
//	GET  /healthz
//	GET  /metrics
//	POST /api/v1/orchestrate      place a new flow
//	POST /api/v1/reroute          force re-evaluation of every active flow
//	GET  /api/v1/flows            active flows and their paths
//	GET  /api/v1/decisions        audit log of routing decisions
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	wLatency     = 1.0
	wLoss        = 220.0
	wRisk        = 140.0
	wLoad        = 45.0
	latencyRefMs = 25.0
	theta        = 0.17 // operating point inherited from the validated predictor
)

// budgetFactor maps service priority to the multiple of theta tolerated on a path.
var budgetFactor = map[int]float64{5: 0.80, 4: 1.00, 3: 1.30, 2: 1.80, 1: 2.50}

type Node struct {
	ID                 string  `json:"id"`
	Type               string  `json:"type"`
	CapacityMbps       float64 `json:"capacity_mbps"`
	LoadMbps           float64 `json:"load_mbps"`
	FailureProbability float64 `json:"failure_probability"`
	Status             string  `json:"status"`
	Telemetry          struct {
		LatencyMs  float64 `json:"latency_ms"`
		PacketLoss float64 `json:"packet_loss"`
	} `json:"telemetry"`
}

type Link struct {
	Source       string  `json:"source"`
	Target       string  `json:"target"`
	PropLatency  float64 `json:"prop_latency_ms"`
	CapacityMbps float64 `json:"capacity_mbps"`
	LoadMbps     float64 `json:"load_mbps"`
}

type Topology struct {
	Nodes []Node `json:"nodes"`
	Links []Link `json:"links"`
}

type ServiceRequest struct {
	FlowID       string  `json:"flow_id"`
	ServiceType  string  `json:"service_type"`
	Priority     int     `json:"priority"`
	Source       string  `json:"source"`
	Destination  string  `json:"destination"`
	DemandMbps   float64 `json:"demand_mbps"`
	MaxLatencyMs float64 `json:"max_latency_ms"`
	MaxLoss      float64 `json:"max_loss"`
}

type RouteDecision struct {
	FlowID           string    `json:"flow_id"`
	ServiceType      string    `json:"service_type"`
	Path             []string  `json:"path"`
	Kind             string    `json:"kind"` // admit | proactive_reroute | reactive_reroute | reject
	PathRisk         float64   `json:"path_risk"`
	BottleneckRisk   float64   `json:"bottleneck_risk"`
	PredictedLatency float64   `json:"predicted_latency_ms"`
	PredictedLoss    float64   `json:"predicted_loss"`
	Budget           float64   `json:"risk_budget"`
	Reason           string    `json:"reason"`
	DecidedAt        time.Time `json:"decided_at"`
}

type Flow struct {
	Request  ServiceRequest `json:"request"`
	Path     []string       `json:"path"`
	Reroutes int            `json:"reroutes"`
}

// ------------------------------------------------------------------ routing

type graph struct {
	nodes map[string]Node
	adj   map[string][]Link
}

func buildGraph(t Topology, excluded map[string]bool, demand float64) *graph {
	g := &graph{nodes: map[string]Node{}, adj: map[string][]Link{}}
	for _, n := range t.Nodes {
		if excluded[n.ID] || n.Status == "failed" {
			continue
		}
		g.nodes[n.ID] = n
	}
	for _, l := range t.Links {
		if _, ok := g.nodes[l.Source]; !ok {
			continue
		}
		if _, ok := g.nodes[l.Target]; !ok {
			continue
		}
		if demand > 0 && l.CapacityMbps-l.LoadMbps < demand {
			continue
		}
		g.adj[l.Source] = append(g.adj[l.Source], l)
		g.adj[l.Target] = append(g.adj[l.Target], Link{
			Source: l.Target, Target: l.Source,
			PropLatency: l.PropLatency, CapacityMbps: l.CapacityMbps, LoadMbps: l.LoadMbps,
		})
	}
	return g
}

// nodePenalty is the per-node term of the composite cost (Eq. 12 of the paper).
func (g *graph) nodePenalty(id string, riskAware bool) float64 {
	n := g.nodes[id]
	util := 0.0
	if n.CapacityMbps > 0 {
		util = n.LoadMbps / n.CapacityMbps
	}
	cost := wLatency*(n.Telemetry.LatencyMs/latencyRefMs) + wLoss*n.Telemetry.PacketLoss
	if util > 0.75 {
		cost += wLoad * (util - 0.75) * (util - 0.75)
	}
	if riskAware {
		cost += wRisk * n.FailureProbability * n.FailureProbability
	}
	return cost
}

// dijkstra returns the least-cost path under the composite metric.
func (g *graph) dijkstra(src, dst string, riskAware bool) []string {
	const inf = math.MaxFloat64
	dist := map[string]float64{}
	prev := map[string]string{}
	visited := map[string]bool{}
	for id := range g.nodes {
		dist[id] = inf
	}
	if _, ok := g.nodes[src]; !ok {
		return nil
	}
	if _, ok := g.nodes[dst]; !ok {
		return nil
	}
	dist[src] = 0
	for {
		best, bestD := "", inf
		for id, d := range dist {
			if !visited[id] && d < bestD {
				best, bestD = id, d
			}
		}
		if best == "" {
			break
		}
		if best == dst {
			break
		}
		visited[best] = true
		for _, l := range g.adj[best] {
			w := wLatency*(l.PropLatency/latencyRefMs) +
				0.5*(g.nodePenalty(best, riskAware)+g.nodePenalty(l.Target, riskAware))
			if nd := bestD + w; nd < dist[l.Target] {
				dist[l.Target] = nd
				prev[l.Target] = best
			}
		}
	}
	if dist[dst] == inf {
		return nil
	}
	path := []string{dst}
	for cur := dst; cur != src; {
		p, ok := prev[cur]
		if !ok {
			return nil
		}
		path = append([]string{p}, path...)
		cur = p
	}
	return path
}

func (g *graph) pathKPI(path []string) (latency, loss, maxRisk, combinedRisk float64) {
	keep := 1.0
	survive := 1.0
	for i, id := range path {
		n := g.nodes[id]
		latency += n.Telemetry.LatencyMs
		keep *= 1 - math.Min(1, n.Telemetry.PacketLoss)
		survive *= 1 - math.Min(0.999, n.FailureProbability)
		if n.FailureProbability > maxRisk {
			maxRisk = n.FailureProbability
		}
		if i > 0 {
			for _, l := range g.adj[path[i-1]] {
				if l.Target == id {
					latency += l.PropLatency
					break
				}
			}
		}
	}
	return latency, 1 - keep, maxRisk, 1 - survive
}

// ------------------------------------------------------------------- service

type orchestrator struct {
	mu        sync.RWMutex
	twinURL   string
	flows     map[string]*Flow
	decisions []RouteDecision
	client    *http.Client
	log       *slog.Logger
	counters  map[string]int
}

func newOrchestrator(twinURL string, log *slog.Logger) *orchestrator {
	return &orchestrator{
		twinURL:  twinURL,
		flows:    map[string]*Flow{},
		client:   &http.Client{Timeout: 3 * time.Second},
		log:      log,
		counters: map[string]int{},
	}
}

func (o *orchestrator) topology() (Topology, error) {
	var t Topology
	resp, err := o.client.Get(o.twinURL + "/api/v1/topology")
	if err != nil {
		return t, err
	}
	defer resp.Body.Close()
	return t, json.NewDecoder(resp.Body).Decode(&t)
}

func (o *orchestrator) place(req ServiceRequest, kind string) RouteDecision {
	top, err := o.topology()
	dec := RouteDecision{
		FlowID: req.FlowID, ServiceType: req.ServiceType, Kind: kind, DecidedAt: time.Now(),
	}
	if err != nil {
		dec.Kind, dec.Reason = "reject", "twin unavailable: "+err.Error()
		return dec
	}
	factor, ok := budgetFactor[req.Priority]
	if !ok {
		factor = 1.0
	}
	budget := math.Min(0.95, factor*theta)
	dec.Budget = budget

	excluded := map[string]bool{}
	limit := math.Min(0.95, 2.5*theta)
	for _, n := range top.Nodes {
		if n.FailureProbability >= math.Max(limit, budget) {
			excluded[n.ID] = true
		}
	}

	g := buildGraph(top, excluded, req.DemandMbps)
	path := g.dijkstra(req.Source, req.Destination, true)
	if path == nil { // relax the hard exclusions before giving up
		g = buildGraph(top, map[string]bool{}, 0)
		path = g.dijkstra(req.Source, req.Destination, true)
		dec.Reason = "relaxed risk exclusion"
	}
	if path == nil {
		dec.Kind, dec.Reason = "reject", "no feasible path"
		o.counters["reject"]++
		return dec
	}
	lat, loss, maxRisk, comb := g.pathKPI(path)
	dec.Path, dec.PredictedLatency, dec.PredictedLoss = path, lat, loss
	dec.BottleneckRisk, dec.PathRisk = maxRisk, comb
	if dec.Reason == "" {
		switch {
		case lat > req.MaxLatencyMs:
			dec.Reason = "latency budget exceeded, best effort"
		case loss > req.MaxLoss:
			dec.Reason = "loss budget exceeded, best effort"
		default:
			dec.Reason = "sla feasible"
		}
	}
	o.counters[kind]++
	return dec
}

func (o *orchestrator) Orchestrate(req ServiceRequest) RouteDecision {
	dec := o.place(req, "admit")
	o.mu.Lock()
	defer o.mu.Unlock()
	if dec.Kind == "admit" {
		o.flows[req.FlowID] = &Flow{Request: req, Path: dec.Path}
	}
	o.decisions = append(o.decisions, dec)
	if len(o.decisions) > 5000 {
		o.decisions = o.decisions[len(o.decisions)-5000:]
	}
	return dec
}

// Reroute re-evaluates every active flow and migrates those whose bottleneck
// risk exceeds their budget or whose path traverses a failed element.
func (o *orchestrator) Reroute() []RouteDecision {
	o.mu.RLock()
	flows := make([]*Flow, 0, len(o.flows))
	for _, f := range o.flows {
		flows = append(flows, f)
	}
	o.mu.RUnlock()
	sort.Slice(flows, func(i, j int) bool {
		return flows[i].Request.Priority > flows[j].Request.Priority
	})

	top, err := o.topology()
	if err != nil {
		return nil
	}
	risk := map[string]float64{}
	failed := map[string]bool{}
	for _, n := range top.Nodes {
		risk[n.ID] = n.FailureProbability
		failed[n.ID] = n.Status == "failed"
	}

	var out []RouteDecision
	for _, f := range flows {
		factor, ok := budgetFactor[f.Request.Priority]
		if !ok {
			factor = 1.0
		}
		budget := math.Min(0.95, factor*theta)
		broken, maxRisk := false, 0.0
		for _, id := range f.Path {
			if failed[id] {
				broken = true
			}
			if risk[id] > maxRisk {
				maxRisk = risk[id]
			}
		}
		if !broken && maxRisk <= budget {
			continue
		}
		kind := "proactive_reroute"
		if broken {
			kind = "reactive_reroute"
		}
		dec := o.place(f.Request, kind)
		if dec.Kind == "reject" || samePath(dec.Path, f.Path) {
			continue
		}
		o.mu.Lock()
		f.Path = dec.Path
		f.Reroutes++
		o.decisions = append(o.decisions, dec)
		o.mu.Unlock()
		out = append(out, dec)
	}
	return out
}

func samePath(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------- HTTP layer

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (o *orchestrator) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /api/v1/orchestrate", func(w http.ResponseWriter, r *http.Request) {
		var req ServiceRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, o.Orchestrate(req))
	})
	mux.HandleFunc("POST /api/v1/reroute", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"decisions": o.Reroute()})
	})
	mux.HandleFunc("GET /api/v1/flows", func(w http.ResponseWriter, r *http.Request) {
		o.mu.RLock()
		defer o.mu.RUnlock()
		writeJSON(w, http.StatusOK, o.flows)
	})
	mux.HandleFunc("GET /api/v1/decisions", func(w http.ResponseWriter, r *http.Request) {
		o.mu.RLock()
		defer o.mu.RUnlock()
		writeJSON(w, http.StatusOK, o.decisions)
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		o.mu.RLock()
		defer o.mu.RUnlock()
		var b strings.Builder
		b.WriteString("# HELP orch_decisions_total Routing decisions by kind\n")
		b.WriteString("# TYPE orch_decisions_total counter\n")
		for k, v := range o.counters {
			fmt.Fprintf(&b, "orch_decisions_total{kind=%q} %d\n", k, v)
		}
		fmt.Fprintf(&b, "orch_active_flows %d\n", len(o.flows))
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(b.String()))
	})
	return mux
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	twin := os.Getenv("TWIN_URL")
	if twin == "" {
		twin = "http://digital-twin-service:8080"
	}
	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8082"
	}
	o := newOrchestrator(twin, log)

	// periodic control loop
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if d := o.Reroute(); len(d) > 0 {
				log.Info("reroute cycle", "migrations", len(d))
			}
		}
	}()

	log.Info("orchestrator-service starting", "addr", addr, "twin", twin)
	srv := &http.Server{Addr: addr, Handler: o.routes(), ReadHeaderTimeout: 5 * time.Second}
	if err := srv.ListenAndServe(); err != nil {
		log.Error("server stopped", "err", err)
		os.Exit(1)
	}
}
