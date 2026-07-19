"""
Internationalization (i18n) module for MARK-XXXIX.
Provides EN/ES string dictionary with a simple key-based lookup.
"""

_current_lang = "en"
__all__ = ["_", "set_language", "on_language_change", "L"]
_callbacks: list = []

LANGUAGES: dict[str, dict[str, str]] = {
    # ── main.py connection strings ──────────────────────────────
    "connecting": {
        "en": "🔌 Connecting...",
        "es": "🔌 Conectando...",
    },
    "connected": {
        "en": "✅ Connected.",
        "es": "✅ Conectado.",
    },
    "reconnecting": {
        "en": "🔄 Reconnecting in 3s...",
        "es": "🔄 Reconectando en 3s...",
    },
    "play_started": {
        "en": "🔊 Play started",
        "es": "🔊 Reproducción iniciada",
    },
    "play_error": {
        "en": "❌ Play: {e}",
        "es": "❌ Error: {e}",
    },
    "connection_error": {
        "en": "⚠️ {e}",
        "es": "⚠️ {e}",
    },
    "shutting_down": {
        "en": "🔴 Shutting down...",
        "es": "🔴 Apagando...",
    },
    "jarvis_online": {
        "en": "SYS: JARVIS online.",
        "es": "SYS: JARVIS en línea.",
    },
    # ── ui.py header ────────────────────────────────────────────
    "mark_xxxix": {
        "en": "MARK XXXIX",
        "es": "MARK XXXIX",
    },
    "jarvis_title": {
        "en": "J.A.R.V.I.S",
        "es": "J.A.R.V.I.S",
    },
    "jarvis_subtitle": {
        "en": "Just A Rather Very Intelligent System",
        "es": "Un Sistema Realmente Muy Inteligente",
    },
    # ── ui.py left panel — sys monitor ──────────────────────────
    "sys_monitor": {
        "en": "◈ SYS MONITOR",
        "es": "◈ MONITOR",
    },
    "cpu": {
        "en": "CPU",
        "es": "CPU",
    },
    "mem": {
        "en": "MEM",
        "es": "MEM",
    },
    "net": {
        "en": "NET",
        "es": "NET",
    },
    "gpu": {
        "en": "GPU",
        "es": "GPU",
    },
    "tmp": {
        "en": "TMP",
        "es": "TMP",
    },
    "uptime": {
        "en": "UP  --:--",
        "es": "ACTIVO  --:--",
    },
    "proc": {
        "en": "PROC  --",
        "es": "PROC  --",
    },
    "os_label": {
        "en": "OS  {os}",
        "es": "SO  {os}",
    },
    "ai_core": {
        "en": "AI CORE\nACTIVE",
        "es": "NÚCLEO IA\nACTIVO",
    },
    "sec_cleared": {
        "en": "SEC\nCLEARED",
        "es": "SEG\nAUTORIZADO",
    },
    "protocol": {
        "en": "PROTOCOL\nXXXVIII",
        "es": "PROTOCOLO\nXXXVIII",
    },
    # ── ui.py right panel ───────────────────────────────────────
    "activity_log": {
        "en": "ACTIVITY LOG",
        "es": "REGISTRO",
    },
    "file_upload": {
        "en": "FILE UPLOAD",
        "es": "SUBIR ARCHIVO",
    },
    "file_hint": {
        "en": "No file loaded — drop or click above to upload",
        "es": "Sin archivo — soltá o hacé clic para subir",
    },
    "file_loaded": {
        "en": "FILE: {name} ({size}) loaded",
        "es": "ARCHIVO: {name} ({size}) cargado",
    },
    "command_input": {
        "en": "COMMAND INPUT",
        "es": "INGRESO DE COMANDO",
    },
    "input_placeholder": {
        "en": "Type a command or question\u2026",
        "es": "Escrib\u00ed un comando o una pregunta\u2026",
    },
    "microphone_active": {
        "en": "\U0001f399  MICROPHONE ACTIVE",
        "es": "\U0001f399  MICR\u00d3FONO ACTIVO",
    },
    "microphone_muted": {
        "en": "\U0001f507  MICROPHONE MUTED",
        "es": "\U0001f507  MICR\u00d3FONO MUDO",
    },
    "fullscreen_btn": {
        "en": "\u26f6  FULLSCREEN  [F11]",
        "es": "\u26f6  PANTALLA COMPLETA  [F11]",
    },
    "mute_log": {
        "en": "SYS: Microphone muted.",
        "es": "SYS: Micr\u00f3fono mudo.",
    },
    "unmute_log": {
        "en": "SYS: Microphone active.",
        "es": "SYS: Micr\u00f3fono activo.",
    },
    "file_tell_jarvis": {
        "en": "Tell JARVIS what to do with it",
        "es": "Decile a JARVIS qu\u00e9 hacer",
    },
    "init_os_log": {
        "en": "SYS: Initialised. OS={os}. JARVIS online.",
        "es": "SYS: Inicializado. SO={os}. JARVIS en l\u00ednea.",
    },
    # ── ui.py footer ────────────────────────────────────────────
    "footer_hotkeys": {
        "en": "[F4] Mute  \u00b7  [F11] Fullscreen",
        "es": "[F4] Silenciar  \u00b7  [F11] Pantalla completa",
    },
    "footer_company": {
        "en": "FatihMakes Industries  \u00b7  MARK XXXIX  \u00b7  CLASSIFIED",
        "es": "FatihMakes Industries  \u00b7  MARK XXXIX  \u00b7  CLASIFICADO",
    },
    "footer_stark": {
        "en": "\u00a9 STARK INDUSTRIES",
        "es": "\u00a9 STARK INDUSTRIES",
    },
    # ── misc ────────────────────────────────────────────────────
    "you_label": {
        "en": "You:",
        "es": "Vos:",
    },
    "language_english": {
        "en": "EN",
        "es": "EN",
    },
    "language_spanish": {
        "en": "ES",
        "es": "ES",
    },
    "file_browse_title": {
        "en": "Select a file for JARVIS",
        "es": "Seleccioná un archivo para JARVIS",
    },
    # ── drop canvas ────────────────────────────────────────────
    "drop_idle": {
        "en": "Drop file here  or  Click to Browse",
        "es": "Solt\u00e1 el archivo o hac\u00e9 clic para buscar",
    },
    "drop_formats": {
        "en": "Images \u00b7 Video \u00b7 Audio \u00b7 PDF \u00b7 Docs \u00b7 Code \u00b7 Data",
        "es": "Im\u00e1genes \u00b7 Video \u00b7 Audio \u00b7 PDF \u00b7 Docs \u00b7 C\u00f3digo \u00b7 Datos",
    },
    "drop_release": {
        "en": "Release to load",
        "es": "Solt\u00e1 para cargar",
    },
    # ── hud status ─────────────────────────────────────────────
    "hud_muted": {
        "en": "MUTED",
        "es": "MUDO",
    },
    "hud_speaking": {
        "en": "SPEAKING",
        "es": "HABLANDO",
    },
    "hud_thinking": {
        "en": "THINKING",
        "es": "PENSANDO",
    },
    "hud_listening": {
        "en": "LISTENING",
        "es": "ESCUCHANDO",
    },
    "hud_processing": {
        "en": "PROCESSING",
        "es": "PROCESANDO",
    },
    # ── setup overlay ──────────────────────────────────────────
    "setup_title": {
        "en": "\u25c8  INITIALISATION REQUIRED",
        "es": "\u25c8  INICIALIZACI\u00d3N REQUERIDA",
    },
    "setup_gemini_key": {
        "en": "GEMINI API KEY",
        "es": "CLAVE API GEMINI",
    },
    "setup_openrouter_key": {
        "en": "OPENROUTER API KEY",
        "es": "CLAVE API OPENROUTER",
    },
    "setup_os": {
        "en": "OPERATING SYSTEM",
        "es": "SISTEMA OPERATIVO",
    },
    "setup_init_btn": {
        "en": "\u25b8  INITIALISE SYSTEMS",
        "es": "\u25b8  INICIALIZAR SISTEMAS",
    },
    "setup_windows": {
        "en": "Windows",
        "es": "Windows",
    },
    "setup_macos": {
        "en": "macOS",
        "es": "macOS",
    },
    "setup_linux": {
        "en": "Linux",
        "es": "Linux",
    },
    "setup_subtitle": {
        "en": "Configure J.A.R.V.I.S. before first boot.",
        "es": "Configur\u00e1 J.A.R.V.I.S. antes del primer inicio.",
    },
    "setup_auto_detected": {
        "en": "Auto-detected: {name}",
        "es": "Detectado: {name}",
    },
    # ── main.py ────────────────────────────────────────────────
    "tool_error_log": {
        "en": "ERR: {tool} \u2014 {msg}",
        "es": "ERR: {tool} \u2014 {msg}",
    },
    "tool_error_speech": {
        "en": "Sir, {tool} encountered an error. {error}",
        "es": "Se\u00f1or, {tool} encontr\u00f3 un error. {error}",
    },
    "safety_filter_blocked": {
        "en": "I'm sorry, I cannot answer that.",
        "es": "Lo siento, no puedo responder eso.",
    },
    "cloud_execution_error": {
        "en": "Sir, I encountered an error executing that request. {error}",
        "es": "Se\u00f1or, encontr\u00e9 un error al ejecutar esa solicitud. {error}",
    },
    "shutdown_log": {
        "en": "SYS: Shutdown requested.",
        "es": "SYS: Apagado solicitado.",
    },
    # ── window title ───────────────────────────────────────────
    "window_title": {
        "en": "J.A.R.V.I.S \u2014 MARK XXXIX",
        "es": "J.A.R.V.I.S \u2014 MARK XXXIX",
    },
}


def set_language(lang: str) -> None:
    """Change the active language and notify callbacks."""
    global _current_lang
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language: {lang}")
    _current_lang = lang
    for cb in _callbacks:
        cb(lang)


def on_language_change(callback) -> None:
    """Register a callback fired when the language changes."""
    _callbacks.append(callback)


def _(key: str, **kwargs) -> str:
    """Look up key in LANGUAGES for the current language.
    
    Falls back to the key itself if not found.
    Supports .format() via kwargs.
    """
    entry = LANGUAGES.get(key)
    if entry is None or not isinstance(entry, dict):
        return key
    text = entry.get(_current_lang, entry.get("en", key))
    if kwargs:
        return text.format(**kwargs)
    return text


# ── Optional autocomplete class ─────────────────────────────────
class L:
    connecting = "connecting"
    connected = "connected"
    reconnecting = "reconnecting"
    play_started = "play_started"
    play_error = "play_error"
    connection_error = "connection_error"
    shutting_down = "shutting_down"
    jarvis_online = "jarvis_online"
    mark_xxxix = "mark_xxxix"
    jarvis_title = "jarvis_title"
    jarvis_subtitle = "jarvis_subtitle"
    sys_monitor = "sys_monitor"
    cpu = "cpu"
    mem = "mem"
    net = "net"
    gpu = "gpu"
    tmp = "tmp"
    uptime = "uptime"
    proc = "proc"
    os_label = "os_label"
    ai_core = "ai_core"
    sec_cleared = "sec_cleared"
    protocol = "protocol"
    activity_log = "activity_log"
    file_upload = "file_upload"
    file_hint = "file_hint"
    file_loaded = "file_loaded"
    command_input = "command_input"
    input_placeholder = "input_placeholder"
    microphone_active = "microphone_active"
    microphone_muted = "microphone_muted"
    fullscreen_btn = "fullscreen_btn"
    mute_log = "mute_log"
    unmute_log = "unmute_log"
    file_tell_jarvis = "file_tell_jarvis"
    init_os_log = "init_os_log"
    footer_hotkeys = "footer_hotkeys"
    footer_company = "footer_company"
    footer_stark = "footer_stark"
    you_label = "you_label"
    language_english = "language_english"
    language_spanish = "language_spanish"
    drop_idle = "drop_idle"
    drop_formats = "drop_formats"
    drop_release = "drop_release"
    hud_muted = "hud_muted"
    hud_speaking = "hud_speaking"
    hud_thinking = "hud_thinking"
    hud_listening = "hud_listening"
    hud_processing = "hud_processing"
    setup_title = "setup_title"
    setup_gemini_key = "setup_gemini_key"
    setup_openrouter_key = "setup_openrouter_key"
    setup_os = "setup_os"
    setup_init_btn = "setup_init_btn"
    setup_windows = "setup_windows"
    setup_macos = "setup_macos"
    setup_linux = "setup_linux"
    tool_error_log = "tool_error_log"
    tool_error_speech = "tool_error_speech"
    safety_filter_blocked = "safety_filter_blocked"
    cloud_execution_error = "cloud_execution_error"
    shutdown_log = "shutdown_log"
    window_title = "window_title"
    setup_subtitle = "setup_subtitle"
    setup_auto_detected = "setup_auto_detected"
