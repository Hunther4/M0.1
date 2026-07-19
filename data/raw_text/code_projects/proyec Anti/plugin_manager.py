import os
import sys
import importlib
import inspect
import logging

logger = logging.getLogger(__name__)

# Global registry for tools
_REGISTRY = {}

# Explicit allowlist of plugin filenames that may be loaded.
# Only files in this set will be imported. Add new plugins here.
ALLOWED_PLUGIN_FILES = frozenset({
    "core_tools.py",
    "web_reader.py",
    "memory_core.py",
    "ast_security_auditor.py",
    "github_diff_auditor.py",
})

# Filenames that shadow stdlib modules — rejected even if in allowlist.
_STDLIB_SHADOWS = frozenset({
    "os.py", "sys.py", "json.py", "re.py", "io.py", "math.py",
    "time.py", "path.py", "collections.py", "functools.py",
    "itertools.py", "subprocess.py", "shutil.py", "glob.py",
    "ast.py", "abc.py", "copy.py", "enum.py", "hashlib.py",
    "logging.py", "random.py", "string.py", "struct.py",
    "typing.py", "unittest.py", "xml.py", "csv.py",
})


def anti_tool(name: str, description: str):
    """
    Decorador para registrar funciones como herramientas de Anti.
    """
    def decorator(func):
        _REGISTRY[name] = {
            "func": func,
            "description": description,
            "name": name
        }
        return func
    return decorator


# Directory for user-created dynamic tools (no allowlist restriction)
DYNAMIC_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "plugins", "user_tools")


class PluginManager:
    """
    Gestor dinámico de plugins. Carga plugins del directorio principal
    (con allowlist) y del subdirectorio user_tools/ (sin allowlist).
    """
    def __init__(self, plugins_dir=None):
        if plugins_dir is None:
            # Resolve relative to this file's location (src/), not CWD
            plugins_dir = os.path.join(os.path.dirname(__file__), "plugins")
        self.plugins_dir = plugins_dir
        self.tools = _REGISTRY
        self.load_plugins()

    def load_plugins(self):
        """Carga plugins desde el directorio principal (allowlist) y user_tools/."""
        base_module = "src.plugins"

        # ── Carga principal (con allowlist) ──
        self._load_from_directory(self.plugins_dir, base_module, use_allowlist=True)

        # ── Carga de herramientas de usuario (sin allowlist) ──
        user_dir = DYNAMIC_TOOLS_DIR
        if os.path.isdir(user_dir):
            self._load_from_directory(user_dir, f"{base_module}.user_tools", use_allowlist=False)

    def _load_from_directory(self, directory: str, base_module: str, use_allowlist: bool):
        """
        Carga todos los .py de un directorio como plugins.
        Si use_allowlist=True, aplica ALLOWED_PLUGIN_FILES + _STDLIB_SHADOWS.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue

            if use_allowlist:
                # Reject stdlib shadows
                if filename in _STDLIB_SHADOWS:
                    logger.warning(
                        f"[PluginManager] Plugin '{filename}' rejected: shadows stdlib module."
                    )
                    continue

                # Enforce allowlist
                if filename not in ALLOWED_PLUGIN_FILES:
                    logger.warning(
                        f"[PluginManager] Plugin '{filename}' rejected: not in ALLOWED_PLUGIN_FILES."
                    )
                    continue

            module_name = f"{base_module}.{filename[:-3]}"
            try:
                if module_name in sys.modules:
                    logger.debug(f"[PluginManager] Módulo {module_name} ya cargado, omitiendo.")
                else:
                    importlib.import_module(module_name)
                    logger.debug(f"[PluginManager] Módulo {module_name} cargado exitosamente.")
            except Exception as e:
                logger.error(f"[PluginManager] Error cargando plugin {filename}: {e}")

    async def execute_tool(self, name: str, raw_args: str):
        """
        Ejecuta una herramienta registrada pasándole los argumentos crudos.
        Soporta de forma nativa tanto funciones síncronas como asíncronas (corutinas).
        """
        if name not in self.tools:
            return f"[ERROR] Herramienta '{name}' no existe o no está registrada."
        
        try:
            func = self.tools[name]["func"]
            if inspect.iscoroutinefunction(func):
                return await func(raw_args)
            return func(raw_args)
        except Exception as e:
            return f"[ERROR] Fallo al ejecutar '{name}': {str(e)}"

    def get_tool_descriptions(self) -> str:
        """
        Devuelve el bloque de texto con las herramientas para inyectar en el prompt.
        """
        if not self.tools:
            return "- No hay herramientas dinámicas cargadas."
            
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- [{name}: argumentos]: {tool['description']}")
        return "\n".join(lines)
