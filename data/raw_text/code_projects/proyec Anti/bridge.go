// Package main — Real-time subprocess bridge between Go TUI and Python agent.
//
// Instead of spawning "python3 main.py" and waiting, this bridge:
//   1. Spawns the Python process with a JSON-line protocol over stdin/stdout
//   2. Streams user input from Go → Python
//   3. Streams agent responses from Python → Go char-by-char (or line-by-line)
//   4. Detects Python process death and restarts transparently
//
// Protocol: JSON lines (newline-delimited JSON)
//   → Go→Python: {"type":"message","content":"hello"}
//   ← Python→Go: {"type":"token","content":"Hel"}
//   ← Python→Go: {"type":"token","content":"lo!"}
//   ← Python→Go: {"type":"done","content":"Hello!"}
//   ← Python→Go: {"type":"error","content":"something broke"}
//   ← Python→Go: {"type":"status","key":"provider","value":"lmstudio"}
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// ── Message Types ─────────────────────────────────────────────────────────

type bridgeMessage struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Key     string `json:"key,omitempty"`
	Value   string `json:"value,omitempty"`
}

type bridgeTokenMsg struct {
	token   string
	done    bool
	err     error
}

type bridgeStatusMsg struct {
	key   string
	value string
}

type bridgePlanMsg struct {
	planJSON string
}

type bridgePlanStepMsg struct {
	stepID string
	status string
	errMsg string
}

type bridgeStartedMsg struct{}
type bridgeStoppedMsg struct{}

// ── Bridge State ─────────────────────────────────────────────────────────

type PythonBridge struct {
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdout   *bufio.Scanner
	mu       sync.Mutex
	running  bool
	stopChan chan struct{}
	wg       sync.WaitGroup
}

var GlobalBridge = &PythonBridge{}

// ── Start Bridge ─────────────────────────────────────────────────────────

func (b *PythonBridge) Start(pythonPath, projectRoot string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.running {
		return nil
	}

	// Resolve Python path
	python := pythonPath
	if python == "" {
		python = "python3"
	}

	// If pythonPath is relative, resolve against projectRoot
	if !filepath.IsAbs(python) && projectRoot != "" {
		if _, err := os.Stat(filepath.Join(projectRoot, python)); err == nil {
			python = filepath.Join(projectRoot, python)
		}
	}

	// Try to find a bridge script or fall back to direct Python
	bridgeScript := filepath.Join(projectRoot, "src", "tui", "bridge.py")
	if _, err := os.Stat(bridgeScript); os.IsNotExist(err) {
		// Fall back: spawn main.py with a special flag
		b.cmd = exec.Command(python, filepath.Join(projectRoot, "main.py"), "--bridge")
	} else {
		b.cmd = exec.Command(python, bridgeScript)
	}

	// Pipes
	stdin, err := b.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("bridge stdin pipe: %w", err)
	}
	b.stdin = stdin

	stdout, err := b.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("bridge stdout pipe: %w", err)
	}
	b.stdout = bufio.NewScanner(stdout)
	// Increase buffer for long responses
	b.stdout.Buffer(make([]byte, 1024*64), 1024*1024)

	b.cmd.Stderr = os.Stderr

	b.stopChan = make(chan struct{})

	if err := b.cmd.Start(); err != nil {
		return fmt.Errorf("bridge start: %w", err)
	}
	b.running = true

	// Read loop
	b.wg.Add(1)
	go b.readLoop()

	return nil
}

func (b *PythonBridge) readLoop() {
	defer b.wg.Done()
	for b.stdout.Scan() {
		line := b.stdout.Text()
		var msg bridgeMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			continue
		}

		switch msg.Type {
		case "token":
			// Stream token to TUI via channel
			select {
			case GlobalEventChan <- bridgeTokenMsg{token: msg.Content}:
			case <-b.stopChan:
				return
			}
		case "done":
			select {
			case GlobalEventChan <- bridgeTokenMsg{done: true, token: msg.Content}:
			case <-b.stopChan:
				return
			}
		case "error":
			select {
			case GlobalEventChan <- bridgeTokenMsg{err: errors.New(msg.Content)}:
			case <-b.stopChan:
				return
			}
		case "status":
			select {
			case GlobalEventChan <- bridgeStatusMsg{key: msg.Key, value: msg.Value}:
			case <-b.stopChan:
				return
			}
		case "plan_create":
			select {
			case GlobalEventChan <- bridgePlanMsg{planJSON: msg.Content}:
			case <-b.stopChan:
				return
			}
		case "plan_step_start", "plan_step_done", "plan_step_fail":
			status := "pending"
			switch msg.Type {
			case "plan_step_start":
				status = "running"
			case "plan_step_done":
				status = "succeeded"
			case "plan_step_fail":
				status = "failed"
			}
			select {
			case GlobalEventChan <- bridgePlanStepMsg{stepID: msg.Content, status: status, errMsg: msg.Value}:
			case <-b.stopChan:
				return
			}
		}
	}

	// Process exited
	b.mu.Lock()
	b.running = false
	b.mu.Unlock()
	GlobalEventChan <- bridgeStoppedMsg{}
}

// ── Send Message ─────────────────────────────────────────────────────────

func (b *PythonBridge) Send(msg bridgeMessage) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.running {
		return fmt.Errorf("bridge not running")
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	_, err = b.stdin.Write(append(data, '\n'))
	return err
}

func (b *PythonBridge) SendMessage(content string) error {
	return b.Send(bridgeMessage{Type: "message", Content: content})
}

// ── Stop Bridge ──────────────────────────────────────────────────────────

func (b *PythonBridge) Stop() {
	b.mu.Lock()
	if !b.running {
		b.mu.Unlock()
		return
	}

	close(b.stopChan)

	if b.stdin != nil {
		b.stdin.Close()
	}

	if b.cmd != nil && b.cmd.Process != nil {
		b.cmd.Process.Kill()
		b.cmd.Wait()
	}

	b.running = false
	b.mu.Unlock()

	b.wg.Wait()
}

// ── Global Event Channel ─────────────────────────────────────────────────

// GlobalEventChan is used by the bridge to push events into the Bubbletea model.
var GlobalEventChan = make(chan interface{}, 100)

// BridgeEventCmd returns a tea.Cmd that reads from GlobalEventChan.
func BridgeEventCmd() tea.Msg {
	select {
	case msg := <-GlobalEventChan:
		return msg
	case <-time.After(100 * time.Millisecond):
		return nil
	}
}

// ── Bridge Chat ──────────────────────────────────────────────────────────

// SendChatMessage bridges user input to the Python agent.
func (m *model) SendChatMessage(msg string) tea.Cmd {
	return func() tea.Msg {
		// Try bridge first
		if GlobalBridge.running {
			if err := GlobalBridge.SendMessage(msg); err == nil {
				return chatStreamStartedMsg{}
			}
		}
		// Fall back to HTTP polling
		return m.sendChatMessageSync(msg)
	}
}

type chatStreamStartedMsg struct{}

// Fallback sync HTTP chat (existing logic)
func (m *model) sendChatMessageSync(msg string) tea.Msg {
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

// ── Chat with streaming ──────────────────────────────────────────────────

// StreamChatUpdate processes incoming bridge tokens for the chat view.
func (m *model) StreamChatUpdate(msg bridgeTokenMsg) {
	if msg.err != nil {
		m.chatHistory = append(m.chatHistory, "Error: "+msg.err.Error())
		m.activeJobId = ""
		m.streamBuffer = ""
		m.refreshChatViewport()
		return
	}

	if msg.done {
		// Final content
		m.streamBuffer += msg.token
		m.chatHistory = append(m.chatHistory, "Anti: "+m.streamBuffer)
		m.activeJobId = ""
		m.streamBuffer = ""
		m.viewingResponse = true
		m.chatInput.Blur()
		m.refreshChatViewport()
		return
	}

	// Streaming token
	m.streamBuffer += msg.token
	m.partialResponse = m.streamBuffer
	m.refreshChatViewport()
}
