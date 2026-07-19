package main

import (
	crypto_rand "crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"context"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Screen types
type Screen int

const (
	ScreenMain Screen = iota
	ScreenSettings
	ScreenSystem
	ScreenDocker
	ScreenDockerLimits
	ScreenKeys
	ScreenModel
	ScreenModelSelection
	ScreenSetup
	ScreenAPIInput
	ScreenTerminal
	ScreenDashboard
	ScreenDockerLogs
	ScreenProjectDir
	ScreenTheme
)

// ── New message types for dashboard, palette, and bridge ──
type switchViewMsg struct {
	view string
}

// Config representation
type Config struct {
	AgentName              string `json:"agent_name"`
	Language               string `json:"language"`
	Personality            string `json:"personality"`
	Provider               string `json:"provider"`
	Model                  string `json:"model"`
	LMStudioURL            string `json:"lm_studio_url"`
	OllamaURL              string `json:"ollama_url"`
	ProjectDir             string `json:"project_dir,omitempty"`
	OpenAIAPIKey           string `json:"openai_api_key,omitempty"`
	DeepSeekAPIKey         string `json:"deepseek_api_key,omitempty"`
	GeminiAPIKey           string `json:"gemini_api_key,omitempty"`
	AnthropicAPIKey        string `json:"anthropic_api_key,omitempty"`
	MinimaxAPIKey          string `json:"minimax_api_key,omitempty"`
	GroqAPIKey             string `json:"groq_api_key,omitempty"`
	OpenRouterAPIKey       string `json:"openrouter_api_key,omitempty"`
	OpenAICompatibleAPIKey string `json:"openaicompatible_api_key,omitempty"`
	OpenAICompatibleURL    string `json:"openaicompatible_url,omitempty"`
	MaxIterations          int    `json:"max_iterations"`
	ReportFormat           string `json:"report_format"`
	Theme                  string `json:"theme,omitempty"`
}

type ModelStatus struct {
	LMStudioOnline   bool
	OllamaOnline     bool
	WorkspaceFiles   int
	EngramsCount     int
	SandboxOnline    bool
	SandboxMemUsedMB int64
	SandboxMemLimMB  int64
	BootEngrams      int
}

type model struct {
	cursor          int
	providerCursor  int
	keysCursor      int
	settingsCursor  int
	systemCursor    int
	dockerCursor    int
	limitsCursor    int
	editLimits      SandboxLimits
	limitsEditing   bool
	limitsInput     textinput.Model
	choices         []string
	config          Config
	status          ModelStatus
	screen          Screen
	selectedAPI     string
	apiInput        textinput.Model
	modelInput       textinput.Model
	chatInput       textinput.Model
	chatHistory     []string
	activeJobId      string
	pollCount        int
	chatViewport     viewport.Model
	viewingResponse  bool
	confirmDeleteLogs bool
	width           int
	height          int
	err             error
	quitting        bool
	pythonPath      string
	projectRoot     string
	metrics         SystemMetrics
	metricsReady    bool
	palette         Palette
	paletteActive   bool
	streamBuffer    string
	partialResponse string
	currentView     string
	chatMsgBuf      []string
	chatBufIdx      int
	dockerLogs       []string
	dockerLogsActive bool
	dockerLogViewport viewport.Model
	projectDirInput  textinput.Model
	activePlan       *TaskPlan
	showPlanSidebar  bool
	dockerLogCancel  context.CancelFunc
	dockerLogFilter   int // 0=all, 1=stdout, 2=stderr
	availableModels   []modelEntry
	fetchingModels    bool
	modelCursor       int
	themeCursor       int
}
type sandboxResetMsg struct {
	err error
}
type sandboxResetResultMsg struct{ err error }
type sandboxStatsMsg struct {
	usedMB int64
	limMB  int64
	online bool
}

type statusData struct {
	LMStudioOnline  bool
	OllamaOnline    bool
	WorkspaceFiles  int
	EngramsCount    int
	BootEngrams     int
	SandboxOnline   bool
}

type statusUpdateMsg struct {
	data statusData
}

// ── Docker log streaming messages ──
type dockerLogLineMsg struct {
	line string
}
type dockerLogsDoneMsg struct{}
type dockerLogsErrMsg struct{ err error }

var dockerLogLines chan string

func (m *model) startDockerLogs() tea.Cmd {
	if m.dockerLogCancel != nil {
		m.dockerLogCancel()
	}

	ctx, cancel := context.WithCancel(context.Background())
	m.dockerLogCancel = cancel

	dockerLogLines = make(chan string, 256)

	go func() {
		_ = StreamDockerLogs(ctx, dockerLogLines)
	}()

	return readNextLogLine()
}

func readNextLogLine() tea.Cmd {
	return func() tea.Msg {
		line, ok := <-dockerLogLines
		if !ok {
			return dockerLogsDoneMsg{}
		}
		return dockerLogLineMsg{line: line}
	}
}

func tickSandboxStats() tea.Cmd {
	return tea.Tick(5*time.Second, func(_ time.Time) tea.Msg {
		used, lim, ok := GetSandboxMemoryMB()
		return sandboxStatsMsg{usedMB: used, limMB: lim, online: ok}
	})
}

func (m *model) sendChatMessage(msg string) tea.Cmd {
	return func() tea.Msg {
		url := "http://localhost:8000/api/chat"
		payload := map[string]string{"message": msg}
		body, _ := json.Marshal(payload)

		resp, err := SignedPost(url, "application/json", body)
		if err != nil {
			return chatErrorMsg{err: err}
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusForbidden {
			return chatErrorMsg{err: fmt.Errorf("403 forbidden: backend rejected HMAC signature")}
		}

		var res map[string]string
		if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
			return chatErrorMsg{err: err}
		}
		jobId := res["job_id"]
		if jobId == "" {
			return chatErrorMsg{err: fmt.Errorf("no job_id in response")}
		}
		return chatJobIdMsg{jobId: jobId}
	}
}

func (m *model) pollJobStatus(jobId string) tea.Cmd {
	return func() tea.Msg {
		url := fmt.Sprintf("http://localhost:8000/api/job/%s", jobId)
		resp, err := SignedGet(url)
		if err != nil {
			return chatErrorMsg{err: err}
		}
		defer resp.Body.Close()

		var res map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
			return chatErrorMsg{err: err}
		}

		status, ok := res["status"].(string)
		if !ok {
			return chatErrorMsg{err: fmt.Errorf("invalid status type")}
		}

		if status == "completed" {
			result, ok := res["result"].(map[string]interface{})
			if !ok {
				return chatErrorMsg{err: fmt.Errorf("invalid result type")}
			}
			resp, ok := result["response"].(string)
			if !ok {
				resp = "No response from backend"
			}
			return chatResponseMsg{response: resp}
		} else if status == "failed" {
			return chatErrorMsg{err: fmt.Errorf("job failed: %v", res["error"])}
		}

		return pollContinueMsg{}
	}
}

type chatJobIdMsg struct{ jobId string }
type chatResponseMsg struct{ response string }
type chatErrorMsg struct{ err error }
type pollContinueMsg struct{}

func doResetSandbox(cwd string) tea.Cmd {
	return func() tea.Msg {
		err := ResetSandbox(cwd)
		return sandboxResetMsg{err: err}
	}
}

func doResetSandboxWithLimits(cwd string, limits SandboxLimits) tea.Cmd {
	return func() tea.Msg {
		err := ResetSandboxWithLimits(cwd, limits)
		return sandboxResetResultMsg{err: err}
	}
}

func initialModel() model {
	ti := textinput.New()
	ti.Placeholder = "Ingresá tu clave aquí..."
	ti.Focus()
	ti.CharLimit = 150
	ti.Width = 35

	cti := textinput.New()
	cti.Placeholder = "Escribí tu mensaje..."
	cti.Focus()
	cti.CharLimit = 500
	cti.Width = 40

	li := textinput.New()
	li.Placeholder = "value"
	li.CharLimit = 10
	li.Width = 10

	mi := textinput.New()
	mi.Placeholder = "model name..."
	mi.CharLimit = 100
	mi.Width = 40

	vp := viewport.New(40, 20)
	vp.SetContent("")

	dockerVP := viewport.New(44, 20)
	dockerVP.SetContent("")

	pdi := textinput.New()
	pdi.Placeholder = "/path/to/your/project"
	pdi.Focus()
	pdi.CharLimit = 200
	pdi.Width = 40

	// Resolve pythonPath relative to the executable's directory, not CWD
	exePath, err := os.Executable()
	projectRoot := "."
	if err == nil {
		projectRoot = filepath.Dir(exePath)
	}
	pythonPath := filepath.Join(projectRoot, "venv", "bin", "python")
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		pythonPath = "python3"
	}

	m := model{
		choices: []string{
			"🚀 Anti Terminal",
			"⚙️  Configuración",
			"🛠️  Sistema",
			"🚪  Salir",
		},
		screen:       ScreenMain,
		apiInput:     ti,
		modelInput:    mi,
		chatInput:    cti,
		chatViewport: vp,
		limitsInput:  li,
		projectDirInput: pdi,
		pythonPath:   pythonPath,
		projectRoot:  projectRoot,
		currentView:  "main",
		chatBufIdx:   -1,
		palette:          NewPalette(),
		dockerLogViewport: dockerVP,
	}

	m.loadConfig()
	return m
}

// resolveConfigPath implements the "Local-First" strategy:
//   1. Prefer config.local.json (gitignored, may contain API keys).
//   2. Fallback to config.json (gitignored, team-shared defaults).
//   3. If neither exists, return an empty string and let the caller surface
//      the actionable error.
func resolveConfigPath() string {
	if _, err := os.Stat("config.local.json"); err == nil {
		return "config.local.json"
	}
	if _, err := os.Stat("config.json"); err == nil {
		return "config.json"
	}
	return ""
}

const configNotFoundMsg = "Configuration file not found. Please copy config.json.example to config.local.json and fill in your keys."

func (m *model) loadConfig() {
	configPath := resolveConfigPath()
	if configPath == "" {
		m.err = errors.New(configNotFoundMsg)
		m.config = Config{
			AgentName:   "Anti",
			Language:    "es",
			Provider:    "auto",
			LMStudioURL: "http://127.0.0.1:1234/v1",
			OllamaURL:   "http://127.0.0.1:11434",
		}
		return
	}

	file, err := os.ReadFile(configPath)
	if err != nil {
		m.err = err
		return
	}

	var cfg Config
	if err := json.Unmarshal(file, &cfg); err != nil {
		m.err = err
		return
	}
	m.config = cfg

	// Apply persisted theme
	if m.config.Theme != "" {
		SetTheme(m.config.Theme)
	}
}

func (m *model) saveConfig() {
	// Always write to the personal/local file so we never clobber shared
	// defaults. Create it from the example if nothing exists yet.
	configPath := "config.local.json"
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		if _, err := os.Stat("config.json"); err == nil {
			// Seed from defaults so the user inherits team-shared values.
			data, readErr := os.ReadFile("config.json")
			if readErr == nil {
				_ = os.WriteFile(configPath, data, 0600)
			}
		}
	}
	data, err := json.MarshalIndent(m.config, "", "  ")
	if err != nil {
		m.err = err
		return
	}
	_ = os.WriteFile(configPath, data, 0600)
}

func (m *model) asyncCheckStatus() tea.Cmd {
	lastNs := lastStatusCheckNs.Load()
	if lastNs != 0 && time.Since(time.Unix(0, lastNs)) < 2*time.Second {
		return nil
	}
	return checkStatusAsync(m.config.LMStudioURL, m.config.OllamaURL)
}

func checkStatusAsync(lmStudioURL, ollamaURL string) tea.Cmd {
	return func() tea.Msg {
		var lmOnline, ollamaOnline, sandboxOnline atomic.Bool
		var workspaceFiles, engramsCount, bootEngrams int
		var wg sync.WaitGroup

		files, _ := filepath.Glob("workspace/*")
		workspaceFiles = len(files)

		dbPath := "memory/cold_archive.db"
		if fi, err := os.Stat(dbPath); err == nil {
			engramsCount = int(fi.Size() / 1024)
		}

		if data, err := os.ReadFile("memory/boot_payload.json"); err == nil {
			var bp struct {
				BootEngramsCount int `json:"boot_engrams_count"`
			}
			if json.Unmarshal(data, &bp) == nil {
				bootEngrams = bp.BootEngramsCount
			}
		}

		wg.Add(3)

		go func() {
			defer wg.Done()
			client := http.Client{Timeout: 750 * time.Millisecond}
			url := lmStudioURL
			if url == "" {
				url = "http://127.0.0.1:1234/v1"
			}
			resp, err := client.Get(url + "/models")
			if err == nil && resp != nil {
				lmOnline.Store(resp.StatusCode == 200)
				_ = resp.Body.Close()
			}
		}()

		go func() {
			defer wg.Done()
			client := http.Client{Timeout: 750 * time.Millisecond}
			url := ollamaURL
			if url == "" {
				url = "http://127.0.0.1:11434"
			}
			resp, err := client.Get(url + "/api/tags")
			if err == nil && resp != nil {
				ollamaOnline.Store(resp.StatusCode == 200)
				_ = resp.Body.Close()
			}
		}()

		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			out, err := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Running}}", "anti-sandbox").Output()
			if err == nil {
				sandboxOnline.Store(strings.TrimSpace(string(out)) == "true")
			}
		}()

		wg.Wait()

		// Stamp throttle after all HTTP work completes so rapid keypresses
		// cannot pile up parallel goroutines.
		lastStatusCheckNs.Store(time.Now().UnixNano())

		return statusUpdateMsg{data: statusData{
			LMStudioOnline: lmOnline.Load(),
			OllamaOnline:   ollamaOnline.Load(),
			WorkspaceFiles: workspaceFiles,
			EngramsCount:   engramsCount,
			BootEngrams:    bootEngrams,
			SandboxOnline:  sandboxOnline.Load(),
		}}
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink,
		tickSandboxStats(),
		m.asyncCheckStatus(),
		collectMetricsAsync(&m.config),
		tickMetrics(),
		pollBridgeEvents(),
	)
}

// pollBridgeEvents returns a tea.Cmd that reads from GlobalEventChan.
// This ensures bridge messages (tokens, plan updates) are dispatched to the Update loop.
func pollBridgeEvents() tea.Cmd {
	return func() tea.Msg {
		select {
		case msg := <-GlobalEventChan:
			return msg
		case <-time.After(100 * time.Millisecond):
			return nil
		}
	}
}

func (m model) handleMainMenuSelection() (tea.Model, tea.Cmd) {
	switch m.cursor {
	case 0: // Anti Terminal
		m.screen = ScreenTerminal
		m.currentView = "terminal"
		return m, nil
	case 1: // Settings
		m.screen = ScreenSettings
		m.settingsCursor = 0
		return m, nil
	case 2: // System
		m.screen = ScreenSystem
		m.systemCursor = 0
		return m, nil
	case 3: // Exit
		m.quitting = true
		return m, tea.Quit
	}
	return m, nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	if m.palette.active {
		if _, ok := msg.(tea.KeyMsg); ok {
			p, paletteCmd := m.palette.Update(msg)
			m.palette = *p
			if !m.palette.active { return m, paletteCmd }
			return m, paletteCmd
		}
		// Fall through — let non-key messages pass
	}

	switch msg := msg.(type) {
	case switchViewMsg:
		switch msg.view {
		case "dashboard":
			m.screen = ScreenDashboard
			return m, collectMetricsAsync(&m.config)
		case "terminal":
			m.screen = ScreenTerminal
		case "main":
			m.screen = ScreenMain
		case "set-project":
			m.screen = ScreenProjectDir
			m.projectDirInput.SetValue(m.config.ProjectDir)
			m.projectDirInput.Focus()
			return m, textinput.Blink
		default:
			m.screen = ScreenMain
		}
		return m, nil
	case sandboxResetResultMsg:
		if msg.err != nil {
			m.err = msg.err
		}
		m.screen = ScreenDocker
		return m, m.asyncCheckStatus()
	case bridgePlanMsg:
		// Parse plan from bridge and set it as active
		plan, err := PlanFromJSON(msg.planJSON)
		if err == nil && plan != nil {
			GlobalPlanManager.SetPlan(plan)
		}
		return m, pollBridgeEvents()
	case bridgePlanStepMsg:
		// Update step status from bridge
		GlobalPlanManager.UpdateStep(msg.stepID, TaskStatus(msg.status), msg.errMsg)
		return m, pollBridgeEvents()
	case bridgeStatusMsg:
		// Handle provider status updates from bridge
		switch msg.key {
		case "provider":
			m.status.LMStudioOnline = msg.value == "lmstudio"
			m.status.OllamaOnline = msg.value == "ollama"
		}
		return m, pollBridgeEvents()
	case nil:
		// Timeout from pollBridgeEvents — nothing happened, keep polling
		return m, pollBridgeEvents()
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Chat in terminal screen: full width minus borders/padding
		m.chatViewport.Width = msg.Width - 10  // generous padding for borders
		if m.chatViewport.Width < 40 { m.chatViewport.Width = 40 }
		m.chatInput.Width = m.chatViewport.Width - 4  // buffer for "❯ " prefix + borders
		if m.chatInput.Width < 20 { m.chatInput.Width = 20 }
		m.chatViewport.Height = msg.Height - 8
		m.refreshChatViewport()
		// Docker logs viewport: mainWidth minus border/padding
		dw := msg.Width - 38 - 2 - 6  // sidebar(38) + spacing(2) + main-border-padding(6)
		if dw < 44 { dw = 44 }
		m.dockerLogViewport.Width = dw
		m.dockerLogViewport.Height = msg.Height - 10
		return m, nil
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			m.quitting = true
			return m, tea.Quit
		}
		if msg.Type == tea.KeyCtrlP { m.palette.Open(); m.palette.prevModel = &m; return m, nil }
		if msg.Type == tea.KeyCtrlT { m.showPlanSidebar = !m.showPlanSidebar; return m, nil }

		switch m.screen {
		case ScreenMain:
			switch msg.Type {
			case tea.KeyEsc:
				m.quitting = true
				return m, tea.Quit
			case tea.KeyUp, tea.KeyDown, tea.KeyRunes:
				isUp := msg.Type == tea.KeyUp || (msg.Type == tea.KeyRunes && len(msg.Runes) > 0 && msg.Runes[0] == 'k')
				isDown := msg.Type == tea.KeyDown || (msg.Type == tea.KeyRunes && len(msg.Runes) > 0 && msg.Runes[0] == 'j')
				if isUp {
					if m.cursor > 0 { m.cursor-- } else { m.cursor = len(m.choices) - 1 }
				} else if isDown {
					if m.cursor < len(m.choices)-1 { m.cursor++ } else { m.cursor = 0 }
				}
			case tea.KeyEnter: return m.handleMainMenuSelection()
			}
		case ScreenTerminal:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenMain
				m.currentView = "main"
				return m, m.asyncCheckStatus()
			case tea.KeyEnter:
				input := m.chatInput.Value()
				if input != "" {
					m.chatMsgBuf = append(m.chatMsgBuf, input)
					m.chatHistory = append(m.chatHistory, "User: "+input)
					m.chatInput.SetValue("")
					m.refreshChatViewport()
					return m, m.sendChatMessage(input)
				}
			case tea.KeyUp:
				if len(m.chatMsgBuf) > 0 && m.chatBufIdx < len(m.chatMsgBuf)-1 {
					m.chatBufIdx++
					m.chatInput.SetValue(m.chatMsgBuf[len(m.chatMsgBuf)-1-m.chatBufIdx])
				}
			case tea.KeyDown:
				if m.chatBufIdx > 0 {
					m.chatBufIdx--
					m.chatInput.SetValue(m.chatMsgBuf[len(m.chatMsgBuf)-1-m.chatBufIdx])
				} else { m.chatBufIdx = -1; m.chatInput.SetValue("") }
			default:
				m.chatInput, cmd = m.chatInput.Update(msg)
				return m, cmd
			}
		case ScreenSettings:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			case tea.KeyUp: if m.settingsCursor > 0 { m.settingsCursor-- }
			case tea.KeyDown: if m.settingsCursor < 2 { m.settingsCursor++ }
			case tea.KeyEnter:
				switch m.settingsCursor {
				case 0: m.screen = ScreenModel
				case 1: m.screen = ScreenKeys
				case 2: m.screen = ScreenTheme; m.themeCursor = 0
				}
			}
		case ScreenKeys:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenSettings
				return m, m.asyncCheckStatus()
			case tea.KeyUp: if m.keysCursor > 0 { m.keysCursor-- }
			case tea.KeyDown: if m.keysCursor < len(apiKeyList(m))-1 { m.keysCursor++ }
			case tea.KeyEnter:
				m.selectedAPI = apiKeyList(m)[m.keysCursor].name
				m.screen = ScreenAPIInput
				m.apiInput.SetValue(apiKeyList(m)[m.keysCursor].key)
				m.apiInput.Focus()
				return m, textinput.Blink
			}
		case ScreenModel:
			switch msg.Type {
			case tea.KeyUp: if m.providerCursor > 0 { m.providerCursor-- } else { m.providerCursor = len(allProviders) - 1 }
			case tea.KeyDown: if m.providerCursor < len(allProviders)-1 { m.providerCursor++ } else { m.providerCursor = 0 }
		case tea.KeyEnter:
			rawProvider := allProviders[m.providerCursor].id
			// Normalize openrouter-free to openrouter for backend compatibility
			if rawProvider == "openrouter-free" {
				m.config.Provider = "openrouter"
			} else {
				m.config.Provider = rawProvider
			}
			m.screen = ScreenModelSelection
			m.fetchingModels = true
			m.availableModels = nil
			m.modelCursor = 0
			return m, fetchModelsCmd(m.config, rawProvider)
			case tea.KeyEsc:
				m.screen = ScreenSettings
				return m, m.asyncCheckStatus()
			}
		case ScreenModelSelection:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenModel
				m.availableModels = nil
				m.fetchingModels = false
				return m, nil
			case tea.KeyUp:
				if !m.fetchingModels && len(m.availableModels) > 0 {
					if m.modelCursor > 0 {
						m.modelCursor--
					} else {
						m.modelCursor = len(m.availableModels) - 1
					}
				}
			case tea.KeyDown:
				if !m.fetchingModels && len(m.availableModels) > 0 {
					if m.modelCursor < len(m.availableModels)-1 {
						m.modelCursor++
					} else {
						m.modelCursor = 0
					}
				}
			case tea.KeyEnter:
				if !m.fetchingModels && len(m.availableModels) > 0 {
					m.config.Model = m.availableModels[m.modelCursor].ID
					m.saveConfig()
					m.screen = ScreenModel
					return m, nil
				}
			}
		case ScreenSystem:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			case tea.KeyUp: if m.systemCursor > 0 { m.systemCursor-- }
			case tea.KeyDown: if m.systemCursor < 3 { m.systemCursor++ }
			case tea.KeyEnter:
				switch m.systemCursor {
				case 0: m.screen = ScreenDocker
				case 1: m.screen = ScreenSetup
				case 2: m.screen = ScreenDashboard; return m, collectMetricsAsync(&m.config)
				}
			}
		case ScreenDocker:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenSystem
				return m, m.asyncCheckStatus()
			case tea.KeyUp: if m.dockerCursor > 0 { m.dockerCursor-- }
			case tea.KeyDown: if m.dockerCursor < 4 { m.dockerCursor++ }
			case tea.KeyEnter:
				switch m.dockerCursor {
				case 1: // Restart
					cwd, _ := os.Getwd()
					return m, doResetSandbox(cwd)
					case 2: // Logs
						m.screen = ScreenDockerLogs
						m.dockerLogsActive = true
						m.dockerLogs = nil
						m.dockerLogViewport.SetContent("")
						return m, m.startDockerLogs()
					case 3: // Limits
					m.screen = ScreenDockerLimits
					m.editLimits = GetSandboxLimits()
					m.limitsCursor = 0
					m.limitsEditing = false
					m.limitsInput.Blur()
				case 4: // Network
					cmd = tea.Println("  🌐 Network config — coming soon")
				}
			}
		case ScreenDockerLogs:
			switch msg.String() {
			case "esc":
				m.dockerLogsActive = false
				m.screen = ScreenDocker
				m.dockerLogs = nil
				return m, nil
			case "0":
				m.dockerLogFilter = 0
				m.dockerLogViewport.SetContent(buildDockerLogContent(m.dockerLogs, m.dockerLogFilter))
			case "1":
				m.dockerLogFilter = 1
				m.dockerLogViewport.SetContent(buildDockerLogContent(m.dockerLogs, m.dockerLogFilter))
			case "2":
				m.dockerLogFilter = 2
				m.dockerLogViewport.SetContent(buildDockerLogContent(m.dockerLogs, m.dockerLogFilter))
			case "r", "R":
				m.dockerLogs = nil
				m.dockerLogViewport.SetContent("")
				return m, m.startDockerLogs()
			case "c", "C":
				m.dockerLogs = nil
				m.dockerLogViewport.SetContent("")
				m.dockerLogViewport.GotoBottom()
			case "up", "k":
				m.dockerLogViewport.LineUp(1)
			case "down", "j":
				m.dockerLogViewport.LineDown(1)
			}
		case ScreenDockerLimits:
			switch msg.Type {
			case tea.KeyEsc:
				if m.limitsEditing {
					m.limitsEditing = false
					m.limitsInput.Blur()
				} else {
					m.screen = ScreenDocker
					return m, m.asyncCheckStatus()
				}
			case tea.KeyUp:
				if !m.limitsEditing && m.limitsCursor > 0 {
					m.limitsCursor--
				}
			case tea.KeyDown:
				if !m.limitsEditing && m.limitsCursor < 3 {
					m.limitsCursor++
				}
			case tea.KeyEnter:
				if m.limitsEditing {
					val := m.limitsInput.Value()
					var err error
					switch m.limitsCursor {
					case 0:
						if v, pErr := strconv.ParseFloat(val, 64); pErr == nil {
							if v < 0.5 || v > 16.0 {
								err = fmt.Errorf("CPUs must be between 0.5 and 16.0")
							} else {
								m.editLimits.CPUs = v
							}
						} else {
							err = fmt.Errorf("Invalid number")
						}
					case 1:
						if v, pErr := strconv.ParseInt(val, 10, 64); pErr == nil {
							if v < 256 || v > 32768 {
								err = fmt.Errorf("Memory must be between 256 and 32768 MB")
							} else {
								m.editLimits.MemoryMB = v
							}
						} else {
							err = fmt.Errorf("Invalid number")
						}
					case 2:
						if v, pErr := strconv.ParseInt(val, 10, 64); pErr == nil {
							if v < 0 || v > m.editLimits.MemoryMB*2 {
								err = fmt.Errorf("Swap must be between 0 and %d MB", m.editLimits.MemoryMB*2)
							} else {
								m.editLimits.SwapMB = v
							}
						} else {
							err = fmt.Errorf("Invalid number")
						}
					}
					if err != nil {
						return m, tea.Println(fmt.Sprintf("  ⚠️ %v, try again", err))
					}
					m.limitsEditing = false
					m.limitsInput.Blur()
				} else if m.limitsCursor < 3 {
					m.limitsEditing = true
					var placeholder string
					switch m.limitsCursor {
					case 0:
						placeholder = fmt.Sprintf("%.1f", m.editLimits.CPUs)
					case 1:
						placeholder = fmt.Sprintf("%d", m.editLimits.MemoryMB)
					case 2:
						placeholder = fmt.Sprintf("%d", m.editLimits.SwapMB)
					}
					m.limitsInput.SetValue(placeholder)
					m.limitsInput.Focus()
				} else {
					cwd, _ := os.Getwd()
					return m, doResetSandboxWithLimits(cwd, m.editLimits)
				}
			default:
				if msg.String() == "p" || msg.String() == "P" {
					m.applyPreset()
				} else if m.limitsEditing {
					m.limitsInput, cmd = m.limitsInput.Update(msg)
				}
			}
		case ScreenTheme:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenSettings
				return m, m.asyncCheckStatus()
			case tea.KeyUp:
				if m.themeCursor > 0 {
					m.themeCursor--
				} else {
					m.themeCursor = len(AllThemes) - 1
				}
			case tea.KeyDown:
				if m.themeCursor < len(AllThemes)-1 {
					m.themeCursor++
				} else {
					m.themeCursor = 0
				}
			case tea.KeyEnter:
				selected := AllThemes[m.themeCursor]
				SetTheme(selected.Name)
				m.config.Theme = selected.Name
				m.saveConfig()
				m.screen = ScreenSettings
				return m, nil
			}
		case ScreenSetup:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenSystem
				return m, m.asyncCheckStatus()
			case tea.KeyEnter:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			}
		case ScreenAPIInput:
			switch msg.Type {
			case tea.KeyEnter:
				key := m.apiInput.Value()
				switch m.selectedAPI {
				case "OpenAI": m.config.OpenAIAPIKey = key
				case "Groq": m.config.GroqAPIKey = key
				case "OpenRouter": m.config.OpenRouterAPIKey = key
				case "DeepSeek": m.config.DeepSeekAPIKey = key
				case "Gemini": m.config.GeminiAPIKey = key
				case "Anthropic": m.config.AnthropicAPIKey = key
				case "MiniMax": m.config.MinimaxAPIKey = key
				case "OpenAI Compatible": m.config.OpenAICompatibleAPIKey = key
				}
				m.saveConfig()
				m.screen = ScreenKeys
				return m, nil
			case tea.KeyEsc:
				m.screen = ScreenKeys
				return m, m.asyncCheckStatus()
			default:
				m.apiInput, cmd = m.apiInput.Update(msg)
				return m, cmd
			}
		case ScreenDashboard:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			case tea.KeyTab:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			}
		case ScreenProjectDir:
			switch msg.Type {
			case tea.KeyEsc:
				m.screen = ScreenMain
				return m, m.asyncCheckStatus()
			case tea.KeyEnter:
				dir := m.projectDirInput.Value()
				if dir == "" {
					return m, tea.Println("  ⚠️ Enter a project directory path")
				}
				// Resolve to absolute path
				if !filepath.IsAbs(dir) {
					if cwd, err := os.Getwd(); err == nil {
						dir = filepath.Join(cwd, dir)
					}
				}
				// Validate directory exists
				if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
					return m, tea.Println(fmt.Sprintf("  ⚠️ Directory not found: %s", dir))
				}
				// Save to config
				m.config.ProjectDir = dir
				m.saveConfig()
				// Scan project in background
				m.screen = ScreenMain
				return m, tea.Batch(
					tea.Println(fmt.Sprintf("  ✅ Project set to: %s", dir)),
					func() tea.Msg {
						_ = RunScanProject(dir)
						return tea.Println("  ✅ Project profile scanned and cached")
					},
				)
			default:
				m.projectDirInput, cmd = m.projectDirInput.Update(msg)
				return m, cmd
			}
		}
	case dockerLogLineMsg:
		if m.screen != ScreenDockerLogs {
			return m, nil
		}
		m.dockerLogs = append(m.dockerLogs, msg.line)
		if len(m.dockerLogs) > 500 {
			m.dockerLogs = m.dockerLogs[100:]
		}
		m.dockerLogViewport.SetContent(buildDockerLogContent(m.dockerLogs, m.dockerLogFilter))
		m.dockerLogViewport.GotoBottom()
		return m, readNextLogLine()
	case dockerLogsDoneMsg:
		if m.screen == ScreenDockerLogs {
			m.dockerLogsActive = false
		}
		return m, nil
	case dockerLogsErrMsg:
		if m.screen == ScreenDockerLogs {
			m.dockerLogs = append(m.dockerLogs, "ERR:error: "+msg.err.Error())
			if len(m.dockerLogs) > 500 {
				m.dockerLogs = m.dockerLogs[100:]
			}
			m.dockerLogViewport.SetContent(buildDockerLogContent(m.dockerLogs, m.dockerLogFilter))
			m.dockerLogViewport.GotoBottom()
			m.dockerLogsActive = false
		}
		return m, nil
	case modelsFetchedMsg:
		m.fetchingModels = false
		if msg.err != nil || len(msg.models) == 0 {
			m.availableModels = nil
		} else {
			sort.Slice(msg.models, func(i, j int) bool {
				return msg.models[i].ID < msg.models[j].ID
			})
			m.availableModels = msg.models
			m.modelCursor = 0
		}
	}
	return m, cmd
}

func (m *model) refreshChatViewport() {
	m.chatViewport.SetContent(m.buildChatContent())
	m.chatViewport.GotoBottom()
}

func (m model) buildChatContent() string {
	var chatArea strings.Builder
	userStyle := AccentStyle()
	antiStyle := lipgloss.NewStyle().Foreground(CurrentTheme.Primary).Bold(true)
	errorStyle := DimStyle()

	for _, msg := range m.chatHistory {
		switch {
		case strings.HasPrefix(msg, "User: "):
			chatArea.WriteString(userStyle.Render("User:") + " " + msg[6:] + "\n\n")
		case strings.HasPrefix(msg, "Anti: "):
			chatArea.WriteString(antiStyle.Render("Anti:") + " " + msg[6:] + "\n\n")
		default:
			chatArea.WriteString(errorStyle.Render(msg) + "\n\n")
		}
	}
	if m.activeJobId != "" {
		chatArea.WriteString(WarningStyle().Render("Anti is thinking... 💭") + "\n")
	}
	return chatArea.String()
}

// managedServerStdin holds the write end of the pipe feeding the managed
// Python server's stdin. The server's umbilical_cord thread monitors this
// fd; if we ever close it, the server emergency-shuts down. We intentionally
// do NOT close it for the lifetime of the TUI.
var managedServerStdin *os.File

// lastStatusCheckNs is a package-level monotonic throttle updated atomically
// AFTER HTTP goroutines complete, preventing request storms on rapid keypresses.
var lastStatusCheckNs atomic.Int64

// startManagedServer spawns the Python backend in ANTI_MANAGED=1 mode,
// generates a fresh 32-byte HMAC secret, writes it to the server's stdin,
// and stores it in the sharedSecret package var so SignedPost can sign
// subsequent requests. The process runs in the background and dies with
// the TUI (Pdeathsig).
func startManagedServer(pythonPath string, projectRoot string) error {
	secret := make([]byte, 32)
	if _, err := crypto_rand.Read(secret); err != nil {
		return fmt.Errorf("generate secret: %w", err)
	}
	SetSharedSecret(secret)

	// If a previous managed server is still running, sever its stdin so
	// the umbilical_cord kills it before we start a new instance.
	if managedServerStdin != nil {
		_ = managedServerStdin.Close()
		managedServerStdin = nil
	}

	cmd := exec.Command(pythonPath, filepath.Join(projectRoot, "server.py"))
	cmd.Env = append(os.Environ(), "ANTI_MANAGED=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGKILL,
	}

	r, w, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("pipe: %w", err)
	}
	cmd.Stdin = r

	var logFile, devNull *os.File
	if ferr := os.MkdirAll("logs", 0755); ferr == nil {
		if f, err := os.OpenFile("logs/server.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
			logFile = f
		}
	}
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	} else {
		if f, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0); err == nil {
			devNull = f
			cmd.Stdout = devNull
			cmd.Stderr = devNull
		}
	}

	if err := cmd.Start(); err != nil {
		_ = r.Close()
		_ = w.Close()
		return fmt.Errorf("start server: %w", err)
	}
	_ = r.Close() // parent doesn't need the read end; child inherited its copy

	// Close parent's copies of stdout/stderr fds (child inherited them)
	if logFile != nil {
		logFile.Close()
	}
	if devNull != nil {
		devNull.Close()
	}

	if _, err := w.Write(secret); err != nil {
		return fmt.Errorf("send secret: %w", err)
	}

	managedServerStdin = w // keep alive: server monitors stdin
	go func() { _ = cmd.Wait() }()

	return nil
}


func (m model) View() string {
	// ── Palette overlay takes full screen ──
	if m.palette.active && m.width > 0 && m.height > 0 {
		return m.palette.View(m.width, m.height)
	}

	// ── Dashboard full-screen View ──
	if m.screen == ScreenDashboard {
		content := renderDashboardView(m, m.width)
		// Add palette hint at bottom
		hint := DimStyle().Render("  Ctrl+P: command palette  •  Tab: switch  •  Esc: back  •  q: quit")
		return lipgloss.JoinVertical(lipgloss.Left, content, "", hint)
	}

	// ── Regular Layout: Sidebar + Main Panel ──

	// Don't render full layout before we know terminal size
	if m.width == 0 {
		return ""
	}

	var mainPanel string
	var sidebarPanel string

	// Calculate dynamic width for main content
	sidebarWidth := 36  // SidebarStyle().Width(36)
	spacing := 2        // "  " separator in JoinHorizontal
	mainWidth := m.width - sidebarWidth - spacing
	if mainWidth < 46 { mainWidth = 46 }  // minimum width

	// Build sidebar using theme styles
	sidebarPanel = renderSidebar(m)

	// Build main panel
	switch m.screen {
	case ScreenMain: mainPanel = renderMainMenu(m, mainWidth)
	case ScreenSettings: mainPanel = renderSettingsScreen(m, mainWidth)
	case ScreenSystem: mainPanel = renderSystemScreen(m, mainWidth)
	case ScreenDocker: mainPanel = renderDockerScreen(m, mainWidth)
	case ScreenDockerLimits: mainPanel = renderDockerLimitsScreen(m, mainWidth)
	case ScreenDockerLogs: mainPanel = renderDockerLogsScreen(m, mainWidth)
	case ScreenKeys: mainPanel = renderKeysScreen(m, mainWidth)
	case ScreenModel: mainPanel = renderModelScreen(m, mainWidth)
	case ScreenModelSelection: mainPanel = renderModelSelectionScreen(m, mainWidth)
	case ScreenSetup: mainPanel = renderSetupScreen(m, mainWidth)
	case ScreenAPIInput: mainPanel = renderAPIInputScreen(m, mainWidth)
	case ScreenProjectDir: mainPanel = renderProjectDirScreen(m, mainWidth)
	case ScreenTerminal: return renderTerminalScreen(m)
	case ScreenTheme: mainPanel = renderThemeScreen(m, mainWidth)
	}

	// Join sidebar + main
	fullView := lipgloss.JoinHorizontal(lipgloss.Top, sidebarPanel, "  ", mainPanel)

	// Title bar — theme-aware
	title := lipgloss.NewStyle().
		Foreground(CurrentTheme.Primary).
		Bold(true).
		Render("  ANTI")
	subtitle := lipgloss.NewStyle().
		Foreground(CurrentTheme.Secondary).
		Italic(true).
		Render("  " + CurrentTheme.Name + " theme  •  Ctrl+P: palette  •  Ctrl+T: tasks")

	return lipgloss.JoinVertical(lipgloss.Left, "", title, subtitle, "", fullView)
}

// renderSidebar moved to sidebar.go

// ── Main Menu ─────────────────────────────────────────────────────────────

func renderSettingsScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("⚙️ CONFIGURACIÓN"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Ajustá la inteligencia, las llaves y el tema\n"))
	
	options := []struct {
		name string
		desc string
	}{
		{"Proveedor & Modelo", "Seleccioná la IA activa"},
		{"Claves API", "Configurá las llaves de acceso"},
		{"Tema", "Cambiá el aspecto visual"},
	}
	
	for i, opt := range options {
		mark := "  "
		style := NormalItemStyle
		if i == m.settingsCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}
		content.WriteString(fmt.Sprintf("  %s%s  %s\n",
			mark,
			style().Render(opt.name),
			DimStyle().Render(opt.desc),
		))
	}
	
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back"))
	
	return MainContentStyle(width).Render(content.String())
}

func renderSystemScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🛠️ SISTEMA"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Control de infraestructura y diagnóstico\n"))
	
	options := []struct {
		name string
		desc string
	}{
		{"Docker Management", "Gestión del sandbox"},
		{"Diagnóstico", "Verificación de dependencias"},
		{"Dashboard", "Métricas en vivo"},
		{"Web Host", "Servidor interactivo"},
	}
	
	for i, opt := range options {
		mark := "  "
		style := NormalItemStyle
		if i == m.systemCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}
		content.WriteString(fmt.Sprintf("  %s%s  %s\n",
			mark,
			style().Render(opt.name),
			DimStyle().Render(opt.desc),
		))
	}
	
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back"))
	
	return MainContentStyle(width).Render(content.String())
}

func renderDockerScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🐳 DOCKER MANAGEMENT"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Sandbox isolation control\n\n"))

	// ── NOOB SECTION ──
	content.WriteString(InfoStyle().Render("⚡ Quick Actions (Noobs)") + "\n")
	statusBadge := ErrorStyle().Render("[OFFLINE]")
	if m.status.SandboxOnline {
		statusBadge = SuccessStyle().Render("[ONLINE]")
	}
	content.WriteString(fmt.Sprintf("  Status: %s\n", statusBadge))
	content.WriteString(fmt.Sprintf("  Action: %s (Enter to Restart)\n", 
		NormalItemStyle().Render(func() string {
			if m.dockerCursor == 1 { return "▸ RESTART" }
			return "  Restart"
		}())))
	content.WriteString("\n")

	// ── PRO SECTION ──
	content.WriteString(InfoStyle().Render("🛠️ Advanced Control (Pros)") + "\n")
	
	proOptions := []struct {
		name string
		desc string
	}{
		{"Real-time Logs", "Stream sandbox stdout/stderr"},
		{"Resource Limits", "Adjust CPU/RAM allocations"},
		{"Network Config", "Modify bridge/port mapping"},
	}

	for i, opt := range proOptions {
		mark := "  "
		style := NormalItemStyle
		// Cursor is offset by 2 (Status + Restart)
		if i+2 == m.dockerCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}
		content.WriteString(fmt.Sprintf("  %s%s %s\n",
			mark,
			style().Render(opt.name),
			DimStyle().Render(opt.desc),
		))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back"))

	return MainContentStyle(width).Render(content.String())
}

func (m *model) applyPreset() {
	presets := []SandboxLimits{
		{CPUs: 2.0, MemoryMB: 2048, SwapMB: 2560}, // Default
		{CPUs: 4.0, MemoryMB: 4096, SwapMB: 5120}, // Performance
		{CPUs: 1.0, MemoryMB: 512, SwapMB: 768},   // Minimal
	}

	var next SandboxLimits
	if m.editLimits.CPUs == 2.0 && m.editLimits.MemoryMB == 2048 {
		next = presets[1]
	} else if m.editLimits.CPUs == 4.0 && m.editLimits.MemoryMB == 4096 {
		next = presets[2]
	} else {
		next = presets[0]
	}
	m.editLimits = next
}

func renderDockerLimitsScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("⚙️ SANDBOX LIMITS"))
	content.WriteString("\n\n")

	fields := []struct {
		label string
		value string
		unit  string
	}{
		{"CPUs", fmt.Sprintf("%.1f", m.editLimits.CPUs), "cores"},
		{"Memory", fmt.Sprintf("%d", m.editLimits.MemoryMB), "MB"},
		{"Swap", fmt.Sprintf("%d", m.editLimits.SwapMB), "MB"},
	}

	for i, f := range fields {
		prefix := "  "
		style := NormalItemStyle
		if i == m.limitsCursor {
			prefix = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}

		valueStr := f.value
		if m.limitsEditing && i == m.limitsCursor {
			valueStr = m.limitsInput.View()
		} else {
			valueStr = style().Render(f.value)
		}

		content.WriteString(fmt.Sprintf("  %s%s: [ %s ] %s\n",
			prefix,
			style().Render(f.label),
			valueStr,
			DimStyle().Render(f.unit),
		))
	}

	content.WriteString("\n")
	if m.limitsCursor == 3 {
		content.WriteString(fmt.Sprintf("  %s\n", SelectedItemStyle().Render("▸ [ Apply & Restart ]")))
	} else {
		content.WriteString(fmt.Sprintf("  %s\n", NormalItemStyle().Render("  [ Apply & Restart ]")))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ select  •  Enter edit  •  Esc back  •  P presets"))

	return MainContentStyle(width).Render(content.String())
}

func buildDockerLogContent(lines []string, filter int) string {
	var sb strings.Builder
	stdoutStyle := lipgloss.NewStyle().Foreground(CurrentTheme.Success)
	stderrStyle := lipgloss.NewStyle().Foreground(CurrentTheme.Error)
	infoStyle := lipgloss.NewStyle().Foreground(CurrentTheme.TextDim)

	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "OUT:"):
			if filter == 0 || filter == 1 {
				sb.WriteString(stdoutStyle.Render(line[4:]) + "\n")
			}
		case strings.HasPrefix(line, "ERR:"):
			if filter == 0 || filter == 2 {
				sb.WriteString(stderrStyle.Render(line[4:]) + "\n")
			}
		default:
			if filter == 0 {
				sb.WriteString(infoStyle.Render(line) + "\n")
			}
		}
	}
	return sb.String()
}

func renderDockerLogsScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("📋 DOCKER LOGS"))
	content.WriteString("\n")

	if len(m.dockerLogs) > 0 {
		content.WriteString(m.dockerLogViewport.View())
	} else {
		statusText := "Waiting for logs..."
		if !m.dockerLogsActive {
			statusText = "Stream stopped"
		}
		content.WriteString(DimStyle().Render("  " + statusText) + "\n")
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  Esc back  •  R refresh  •  C clear  •  0 all  •  1 out  •  2 err"))

	return MainContentStyle(width).Render(content.String())
}

func renderMainMenu(m model, width int) string {
	var menu strings.Builder
	menu.WriteString(TitleStyle().Render("MAIN MENU"))
	menu.WriteString("\n")

	for i, choice := range m.choices {
		cursor := "  "
		if i == m.cursor {
			cursor = AccentStyle().Render("▸")
			rendered := SelectedItemStyle().Render(fmt.Sprintf(" %s", choice))
			menu.WriteString(fmt.Sprintf("  %s %s\n", cursor, rendered))
		} else {
			rendered := NormalItemStyle().Render(fmt.Sprintf("  %s", choice))
			menu.WriteString(fmt.Sprintf("  %s  %s\n", cursor, rendered))
		}
	}

	menu.WriteString("\n")
	menu.WriteString(DimStyle().Render("  ↑↓/jk navigate  •  Enter select  •  Esc/Ctrl+C quit"))

	return MainContentStyle(width).Render(menu.String())
}

// ── Keys Screen ───────────────────────────────────────────────────────────

type apiKeyEntry struct {
	id   string
	name string
	key  string // the value
}

func apiKeyList(m model) []apiKeyEntry {
	return []apiKeyEntry{
		{"openai", "OpenAI", m.config.OpenAIAPIKey},
		{"groq", "Groq", m.config.GroqAPIKey},
		{"openrouter", "OpenRouter", m.config.OpenRouterAPIKey},
		{"deepseek", "DeepSeek", m.config.DeepSeekAPIKey},
		{"gemini", "Gemini", m.config.GeminiAPIKey},
		{"anthropic", "Anthropic", m.config.AnthropicAPIKey},
		{"minimax", "MiniMax", m.config.MinimaxAPIKey},
		{"openaicompatible", "OpenAI Compatible", m.config.OpenAICompatibleAPIKey},
	}
}

func renderKeysScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🔌 API KEYS"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter configure  •  Esc back\n\n"))

	keys := apiKeyList(m)
	for i, k := range keys {
		status := "○"
		statusStyle := DimStyle()
		if k.key != "" {
			status = "●"
			statusStyle = SuccessStyle()
		}

		mark := "  "
		style := NormalItemStyle
		if i == m.keysCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}
		content.WriteString(fmt.Sprintf("  %s%s %s\n",
			mark,
			style().Render(k.name),
			statusStyle.Render(status),
		))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter configure  •  Esc back"))

	return MainContentStyle(width).Render(content.String())
}

// ── Model Screen ──────────────────────────────────────────────────────────

type providerEntry struct {
	id   string
	name string
	desc string
}

var allProviders = []providerEntry{
	{"auto", "Auto-detect", "Local LM Studio → Ollama"},
	{"lmstudio", "LM Studio", "Local ⚡"},
	{"ollama", "Ollama", "Local ⚡"},
	{"openai", "OpenAI", "Cloud ☁️"},
	{"groq", "Groq", "Cloud ☁️"},
	{"openrouter-free", "OpenRouter (Free)", "Free models"},
	{"openrouter", "OpenRouter (All)", "All models"},
	{"gemini", "Gemini", "Cloud ☁️"},
	{"deepseek", "DeepSeek", "Cloud ☁️"},
	{"anthropic", "Anthropic Claude", "Cloud ☁️"},
	{"minimax", "MiniMax", "Cloud ☁️"},
	{"openaicompatible", "OpenAI Compatible", "Hybrid 🔌"},
}

func renderModelScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🤖 SELECT PROVIDER"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back\n\n"))

	// Calculate max name length for alignment
	maxNameLen := 0
	for _, p := range allProviders {
		nameLen := len(p.name)
		if nameLen > maxNameLen {
			maxNameLen = nameLen
		}
	}

	for i, p := range allProviders {
		mark := "  "
		style := NormalItemStyle
		if i == m.providerCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}

		// Active mark: show for exact match OR when provider starts with same prefix (for OpenRouter free/all)
		activeMark := ""
		if m.config.Provider == p.id ||
			(strings.HasPrefix(m.config.Provider, "openrouter") && strings.HasPrefix(p.id, "openrouter")) {
			activeMark = SuccessStyle().Render(" ●")
		}

		// Pad name to maxNameLen for alignment
		paddedName := p.name + strings.Repeat(" ", maxNameLen-len(p.name))

		content.WriteString(fmt.Sprintf("  %s%s%s  %s\n",
			mark,
			style().Render(paddedName),
			activeMark,
			DimStyle().Render(p.desc),
		))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back"))

	return MainContentStyle(width).Render(content.String())
}

func renderModelSelectionScreen(m model, width int) string {
	var content strings.Builder
	provName := m.config.Provider
	for _, p := range allProviders {
		if p.id == provName {
			provName = p.name
			break
		}
	}
	content.WriteString(TitleStyle().Render("🎯 SELECT MODEL"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Provider: " + provName + "\n\n"))

	if m.fetchingModels {
		content.WriteString(InfoStyle().Render("  ⏳ Fetching available models...") + "\n")
	} else if len(m.availableModels) > 0 {
		for i, model := range m.availableModels {
			cursor := "  "
			if i == m.modelCursor {
				cursor = AccentStyle().Render("▸")
				content.WriteString(fmt.Sprintf("  %s %s\n", cursor, SelectedItemStyle().Render(model.Name)))
			} else {
				content.WriteString(fmt.Sprintf("  %s %s\n", cursor, NormalItemStyle().Render(model.Name)))
			}
		}
	} else {
		// Fetch failed or no models — show error, NO text input fallback
		content.WriteString(ErrorStyle().Render("  ✗ Could not fetch models") + "\n")
		content.WriteString(DimStyle().Render("  Make sure your API key is valid and the") + "\n")
		content.WriteString(DimStyle().Render("  provider is reachable, then try again.") + "\n")
	}

	content.WriteString("\n")
	if m.fetchingModels {
		content.WriteString(DimStyle().Render("  Esc: cancel"))
	} else if len(m.availableModels) > 0 {
		content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter select  •  Esc back"))
	} else {
		content.WriteString(DimStyle().Render("  Esc: back"))
	}

	return MainContentStyle(width).Render(content.String())
}

// ── Theme Screen ──────────────────────────────────────────────────────────

func renderThemeScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🎨 TEMA"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Elegí el aspecto visual del TUI\n"))

	for i, t := range AllThemes {
		mark := "  "
		style := NormalItemStyle
		if i == m.themeCursor {
			mark = SelectedItemStyle().Render("▸ ")
			style = SelectedItemStyle
		}

		// Show a small color swatch using the theme's Primary color
		swatch := lipgloss.NewStyle().
			Foreground(lipgloss.Color(t.Primary)).
			Render("■■■")

		// Mark the active theme
		activeMark := ""
		if CurrentTheme.Name == t.Name {
			activeMark = "  " + SuccessStyle().Render("● active")
		}

		content.WriteString(fmt.Sprintf("  %s%s  %s%s\n",
			mark,
			style().Render(t.Name),
			swatch,
			activeMark,
		))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  ↑/↓ navigate  •  Enter apply  •  Esc back"))

	return MainContentStyle(width).Render(content.String())
}

// ── Setup Screen ──────────────────────────────────────────────────────────

func renderSetupScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("⚙️  SYSTEM SETUP"))
	content.WriteString("\n")
	content.WriteString(DimStyle().Render("Diagnostics & configuration:\n\n"))

	checks := []struct {
		label string
		ok    bool
	}{
		{"Docker installed", func() bool {
			_, err := exec.LookPath("docker")
			return err == nil
		}()},
		{"config.local.json", func() bool {
			_, err := os.Stat("config.local.json")
			return err == nil
		}()},
		{"workspace/ directory", func() bool {
			fi, err := os.Stat("workspace")
			return err == nil && fi.IsDir()
		}()},
		{"memory/ directory", func() bool {
			fi, err := os.Stat("memory")
			return err == nil && fi.IsDir()
		}()},
		{"anti-sandbox container", m.status.SandboxOnline},
		{"LLM backend online", m.status.LMStudioOnline || m.status.OllamaOnline},
		{"Python dependencies", func() bool {
			_, err := exec.LookPath(m.pythonPath)
			return err == nil
		}()},
	}

	for _, c := range checks {
		content.WriteString(fmt.Sprintf("  %s  %s\n",
			ColoredLabel(c.label, c.ok),
			DimStyle().Render(renderCheckStatus(c.ok)),
		))
	}

	content.WriteString("\n")
	content.WriteString(DimStyle().Render("  Enter/Esc: back to System"))

	return MainContentStyle(width).Render(content.String())
}

func renderCheckStatus(ok bool) string {
	if ok {
		return ""
	}
	return "• Missing"
}

// ── API Input Screen ─────────────────────────────────────────────────────

func renderAPIInputScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("🔑 API KEY"))
	content.WriteString("\n\n")
	content.WriteString(DimStyle().Render(fmt.Sprintf("Enter key for: %s\n\n", m.selectedAPI)))
	content.WriteString(m.apiInput.View())
	content.WriteString("\n\n")
	content.WriteString(DimStyle().Render("  Enter: save  •  Esc: cancel"))

	return MainContentStyle(width).Render(content.String())
}

func renderProjectDirScreen(m model, width int) string {
	var content strings.Builder
	content.WriteString(TitleStyle().Render("📁 SET PROJECT DIRECTORY"))
	content.WriteString("\n\n")
	content.WriteString(DimStyle().Render("Anti will scan this directory to understand your project.\n"))
	content.WriteString(DimStyle().Render("Detects language, framework, structure, and dependencies.\n\n"))

	if m.config.ProjectDir != "" {
		content.WriteString(SuccessStyle().Render("Current: "))
		content.WriteString(m.config.ProjectDir)
		content.WriteString("\n\n")
	}

	content.WriteString(DimStyle().Render("Enter new path:\n\n"))
	content.WriteString(m.projectDirInput.View())
	content.WriteString("\n\n")
	content.WriteString(DimStyle().Render("  Enter: scan & save  •  Esc: cancel"))

	return MainContentStyle(width).Render(content.String())
}

func renderTerminalScreen(m model) string {
	plan := GlobalPlanManager.GetPlan()
	hasPlan := plan != nil && len(plan.Steps) > 0
	showSidebar := hasPlan && m.showPlanSidebar

	// ── Width Calculations ──
	sidebarWidth := 34 // Including borders and padding
	mainWidth := m.width - 4
	if showSidebar {
		mainWidth = m.width - sidebarWidth - 4 // 4 for the gap and borders
	}

	// ── Header: Quick Metrics ──
	// We constrain the header to mainWidth so it doesn't push the sidebar
	header := lipgloss.NewStyle().Width(mainWidth).Render(
		lipgloss.JoinHorizontal(lipgloss.Top,
			TitleStyle().Render("🚀 ANTI TERMINAL"),
			DimStyle().Render(fmt.Sprintf("  |  Model: %s  |  CPU: %.0f cores  |  RAM: %.0f%%", 
				m.config.Model, m.metrics.CPUPercent, m.metrics.MemPercent)),
		),
	)

	// ── Main Body: Chat Viewport ──
	chatView := lipgloss.NewStyle().
		Background(CurrentTheme.Surface).
		Padding(1, 2).
		Width(mainWidth).
		Height(m.height - 8).
		Render(m.chatViewport.View())

	// ── Footer: Input ──
	inputRow := lipgloss.JoinHorizontal(lipgloss.Left,
		AccentStyle().Render("❯ "),
		m.chatInput.View(),
	)
	
	footer := lipgloss.NewStyle().Width(mainWidth).Render(
		lipgloss.JoinVertical(lipgloss.Left,
			inputRow,
			DimStyle().Render("  Enter: send  •  ↑↓ history  •  Esc: menu  •  Ctrl+P palette  •  Ctrl+T tasks"),
		),
	)

	// ── Main Column ──
	mainColumn := lipgloss.JoinVertical(lipgloss.Left,
		header,
		"",
		chatView,
		"",
		footer,
	)

	// ── Task Bar Sidebar ──
	if showSidebar {
		taskBar := RenderTaskBar(30)
		return lipgloss.JoinHorizontal(lipgloss.Top,
			mainColumn,
			" ",
			taskBar,
		)
	}

	return mainColumn
}

// ── Chat Screen ──────────────────────────────────────────────────────────


func main() {
	if exePath, err := os.Executable(); err == nil {
		appDir := filepath.Dir(exePath)
		_ = os.Chdir(appDir)
	}

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "--mem-init":
			if err := RunMemBoot(); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		case "--mem-search":
			if len(os.Args) < 3 {
				fmt.Println("[]")
				os.Exit(1)
			}
			if err := RunMemSearch(os.Args[2]); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		case "--mem-get":
			if len(os.Args) < 3 {
				fmt.Println("{\"status\": \"error\", \"message\": \"Falta id de engram\"}")
				os.Exit(1)
			}
			if err := RunMemGet(os.Args[2]); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		case "--mem-distill":
			if err := RunMemDistill(); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		case "--mem-reinforce":
			if len(os.Args) < 3 {
				fmt.Println("{\"status\": \"error\", \"message\": \"Falta id de engram\"}")
				os.Exit(1)
			}
			if err := RunMemReinforce(os.Args[2]); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		case "--scan-project":
			dir := "."
			if len(os.Args) >= 3 {
				dir = os.Args[2]
			}
			if err := RunScanProject(dir); err != nil {
				fmt.Printf("{\"status\": \"error\", \"message\": \"%v\"}\n", err)
				os.Exit(1)
			}
			os.Exit(0)
		default:
			fmt.Printf("Comando desconocido: %s\nUsar: --mem-init, --mem-search, --mem-get, --mem-distill, --mem-reinforce\n", os.Args[1])
			os.Exit(1)
		}
	}

	// Clean up sandbox container when the TUI exits
	defer StopSandbox()

	p := tea.NewProgram(initialModel(), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Ocurrió un error en el TUI: %v", err)
		os.Exit(1)
	}
}
