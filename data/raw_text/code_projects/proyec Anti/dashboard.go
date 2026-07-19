// Package main — Live system dashboard with parallel metric collection.
//
// Every 3 seconds, a goroutine pool collects: CPU, RAM, Docker sandbox,
// FTS5 engram count, workspace file count, and provider status.
// The dashboard renders all metrics in a compact, beautiful layout.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	// Reuse the existing sandbox stats
	_ "github.com/docker/docker/client"
)

// ── Metric Types ──────────────────────────────────────────────────────────

type SystemMetrics struct {
	CPUPercent     float64
	MemUsedMB      uint64
	MemTotalMB     uint64
	MemPercent     float64
	GoRoutines     int
	WorkspaceFiles int
	EngramsCount   int
	BootEngrams    int

	LMStudioOnline  bool
	OllamaOnline    bool
	SandboxOnline   bool
	SandboxMemUsed  int64
	SandboxMemLimit int64
	ServerOnline    bool // localhost:8000 health

	Provider string
	Model    string
}

type metricsTickMsg struct{}

type metricsUpdateMsg struct {
	SystemMetrics
}

// ── Metric Collection ────────────────────────────────────────────────────

func tickMetrics() tea.Cmd {
	return tea.Tick(3*time.Second, func(_ time.Time) tea.Msg {
		return metricsTickMsg{}
	})
}

func collectMetricsAsync(cfg *Config) tea.Cmd {
	return func() tea.Msg {
		var m SystemMetrics
		var wg sync.WaitGroup

		// ── CPU: number of procs (quick, no cgo needed) ──
		m.CPUPercent = float64(runtime.NumCPU())

		// ── Memory: read /proc/meminfo ──
		if data, err := os.ReadFile("/proc/meminfo"); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "MemTotal:") {
					fmt.Sscanf(line, "MemTotal: %d kB", &m.MemTotalMB)
					m.MemTotalMB /= 1024
				}
				if strings.HasPrefix(line, "MemAvailable:") {
					var availKB uint64
					fmt.Sscanf(line, "MemAvailable: %d kB", &availKB)
					m.MemUsedMB = m.MemTotalMB - availKB/1024
				}
			}
			if m.MemTotalMB > 0 {
				m.MemPercent = float64(m.MemUsedMB) / float64(m.MemTotalMB) * 100
			}
		}

		m.GoRoutines = runtime.NumGoroutine()

		// ── Workspace files ──
		files, _ := filepath.Glob("workspace/*")
		m.WorkspaceFiles = len(files)

		// ── Engram count (KB from cold_archive.db) ──
		if fi, err := os.Stat("memory/cold_archive.db"); err == nil {
			m.EngramsCount = int(fi.Size() / 1024)
		}

		// ── Boot engrams ──
		if data, err := os.ReadFile("memory/boot_payload.json"); err == nil {
			var bp struct {
				BootEngramsCount int `json:"boot_engrams_count"`
			}
			if json.Unmarshal(data, &bp) == nil {
				m.BootEngrams = bp.BootEngramsCount
			}
		}

		// ── Config from model ──
		if cfg != nil {
			m.Provider = cfg.Provider
			m.Model = cfg.Model
		}

		// ── Parallel checks ──
		wg.Add(4)
		lmURL := "http://127.0.0.1:1234/v1"
		olURL := "http://127.0.0.1:11434"
		if cfg != nil {
			if cfg.LMStudioURL != "" {
				lmURL = cfg.LMStudioURL
			}
			if cfg.OllamaURL != "" {
				olURL = cfg.OllamaURL
			}
		}

		var lmOn, olOn, sbOn, srvOn atomic.Bool
		var sbUsed, sbLim atomic.Int64

		go func() {
			defer wg.Done()
			client := http.Client{Timeout: 750 * time.Millisecond}
			if resp, err := client.Get(lmURL + "/models"); err == nil && resp != nil {
				lmOn.Store(resp.StatusCode == 200)
				resp.Body.Close()
			}
		}()

		go func() {
			defer wg.Done()
			client := http.Client{Timeout: 750 * time.Millisecond}
			if resp, err := client.Get(olURL + "/api/tags"); err == nil && resp != nil {
				olOn.Store(resp.StatusCode == 200)
				resp.Body.Close()
			}
		}()

		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if out, err := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Running}}", "anti-sandbox").Output(); err == nil {
				sbOn.Store(strings.TrimSpace(string(out)) == "true")
			}
			used, lim, ok := GetSandboxMemoryMB()
			if ok {
				sbUsed.Store(used)
				sbLim.Store(lim)
			}
		}()

		go func() {
			defer wg.Done()
			client := http.Client{Timeout: 500 * time.Millisecond}
			resp, err := client.Get("http://localhost:8000/health")
			if err == nil && resp != nil {
				srvOn.Store(resp.StatusCode == 200)
				resp.Body.Close()
			}
		}()

		wg.Wait()
		m.LMStudioOnline = lmOn.Load()
		m.OllamaOnline = olOn.Load()
		m.SandboxOnline = sbOn.Load()
		m.SandboxMemUsed = sbUsed.Load()
		m.SandboxMemLimit = sbLim.Load()
		m.ServerOnline = srvOn.Load()
		return metricsUpdateMsg{SystemMetrics: m}
	}
}

// ── Dashboard Rendering ───────────────────────────────────────────────────

func renderDashboard(m SystemMetrics) string {
	var b strings.Builder

	// ── System Section ──
	b.WriteString("  " + InfoStyle().Render("⚡ SYSTEM") + "\n")

	// CPU + Go routines
	cpuStr := fmt.Sprintf("%.0f cores", math.Round(m.CPUPercent))
	b.WriteString(renderMetricLine("CPU", cpuStr))
	b.WriteString(renderMetricLine("Goroutines", fmt.Sprintf("%d", m.GoRoutines)))

	// Memory bar
	memLabel := fmt.Sprintf("%d MB / %d MB", m.MemUsedMB, m.MemTotalMB)
	memBar := ProgressBarStyle(int(m.MemUsedMB), int(m.MemTotalMB))
	if memBar != "" {
		b.WriteString(renderMetricLine("RAM", memBar+"  "+memLabel))
	} else {
		b.WriteString(renderMetricLine("RAM", memLabel))
	}

	// ── Storage Section ──
	b.WriteString("\n")
	b.WriteString("  " + InfoStyle().Render("💾 STORAGE") + "\n")

	b.WriteString(renderMetricLine("Workspace", fmt.Sprintf("%d files", m.WorkspaceFiles)))
	b.WriteString(renderMetricLine("Knowledge", fmt.Sprintf("%d KB", m.EngramsCount)))
	if m.BootEngrams > 0 {
		b.WriteString(renderMetricLine("Boot", fmt.Sprintf("%d engrams", m.BootEngrams)))
	}

	// ── Services Section ──
	b.WriteString("\n")
	b.WriteString("  " + InfoStyle().Render("🔌 SERVICES") + "\n")

	b.WriteString(renderMetricLine("LM Studio", boolToStatus(m.LMStudioOnline)))
	b.WriteString(renderMetricLine("Ollama", boolToStatus(m.OllamaOnline)))
	b.WriteString(renderMetricLine("Web Server", boolToStatus(m.ServerOnline)))

	// ── Docker Sandbox ──
	b.WriteString("\n")
	b.WriteString("  " + InfoStyle().Render("🐳 SANDBOX") + "\n")

	b.WriteString(renderMetricLine("Status", boolToStatus(m.SandboxOnline)))
	if m.SandboxOnline && m.SandboxMemLimit > 0 {
		sandboxBar := ProgressBarStyle(int(m.SandboxMemUsed), int(m.SandboxMemLimit))
		sbLabel := fmt.Sprintf("%d MB / %d MB", m.SandboxMemUsed, m.SandboxMemLimit)
		if sandboxBar != "" {
			b.WriteString(renderMetricLine("Memory", sandboxBar+"  "+sbLabel))
		} else {
			b.WriteString(renderMetricLine("Memory", sbLabel))
		}
	}

	// ── Config ──
	if m.Provider != "" {
		b.WriteString("\n")
		b.WriteString("  " + InfoStyle().Render("⚙️  CONFIG") + "\n")
		b.WriteString(renderMetricLine("Provider", m.Provider))
		if m.Model != "" {
			b.WriteString(renderMetricLine("Model", m.Model))
		}
	}

	return b.String()
}

func renderMetricLine(label, value string) string {
	return fmt.Sprintf("    %-12s %s\n", label+":", value)
}

func boolToStatus(ok bool) string {
	if ok {
		return "● Online"
	}
	return "○ Offline"
}

// ── Dashboard View (full screen wrapper) ─────────────────────────────────

func renderDashboardView(m model, width int) string {
	content := renderDashboard(m.metrics)

	header := TitleStyle().Render("📊 ANTI DASHBOARD")
	subtitle := SubtitleStyle().Render("Live System Metrics  •  Every 3s")
	help := DimStyle().Render("  Tab: switch view  •  Esc: main menu  •  q: quit")

	return lipgloss.JoinVertical(
		lipgloss.Left,
		header,
		subtitle,
		"",
		content,
		"",
		help,
	)
}
