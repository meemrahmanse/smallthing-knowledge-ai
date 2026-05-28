from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable
from enum import Enum


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class Theme(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.GROQ
    model: str = "mixtral-8x7b-32768"
    temperature: float = 0.3
    maxtokens: int = 4096
    topp: float = 0.9
    groqapikey: str = ""
    openaiapikey: str = ""
    anthropicapikey: str = ""
    ollamaurl: str = "http://localhost:11434/api/chat"


@dataclass
class RAGConfig:
    chunksize: int = 800
    chunkoverlap: int = 100
    topkresults: int = 15
    similaritythreshold: float = 0.45
    rerankenabled: bool = True
    reranktopk: int = 12
    searchstrategy: str = "hybrid"
    keywordboost: float = 0.2
    contextexpansion: bool = True


@dataclass
class UIConfig:
    theme: Theme = Theme.DARK
    language: str = "it"
    showsources: bool = True
    showconfidence: bool = True
    showtiming: bool = False
    streamingenabled: bool = True
    quicksuggestions: list = field(default_factory=lambda: ["Lista tutti i fix", "Tutti gli ENV", "NPC in Room019"]) 


@dataclass
class SecurityConfig:
    apikeyrequired: bool = False
    ratelimitrequests: int = 60
    ratelimitwindow: int = 60
    maxfilesizemb: int = 50
    allowedextensions: list = field(default_factory=lambda: [".pdf", ".txt", ".md", ".docx", ".py", ".cs", ".js", ".json"])


@dataclass
class SystemConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    version: str = "4.0.0"
    lastmodified: str = ""
    modifiedby: str = ""


class ConfigManager:
    def __init__(self, configpath: Path = Path("config/settings.json")):
        self.configpath = configpath
        self.configpath.parent.mkdir(parents=True, exist_ok=True)
        self._config: SystemConfig = SystemConfig()
        self.lock = threading.RLock()
        self.listeners: list[Callable[[str, Any, Any], None]] = []
        self.load()

    def load(self):
        if not self.configpath.exists():
            self.save()
            return
        try:
            with open(self.configpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._config = SystemConfig(
                llm=LLMConfig(**data.get('llm', {})),
                rag=RAGConfig(**data.get('rag', {})),
                ui=UIConfig(**data.get('ui', {})),
                security=SecurityConfig(**data.get('security', {})),
                version=data.get('version', '4.0.0'),
                lastmodified=data.get('lastmodified', ''),
                modifiedby=data.get('modifiedby', '')
            )
        except Exception:
            self._config = SystemConfig()

    def save(self):
        self._config.lastmodified = datetime.now().isoformat()
        data = {
            'llm': asdict(self._config.llm),
            'rag': asdict(self._config.rag),
            'ui': asdict(self._config.ui),
            'security': asdict(self._config.security),
            'version': self._config.version,
            'lastmodified': self._config.lastmodified,
            'modifiedby': self._config.modifiedby,
        }
        with open(self.configpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def config(self) -> SystemConfig:
        with self.lock:
            return self._config

    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split('.')
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj

    def set(self, path: str, value: Any, modifiedby: str = 'system') -> bool:
        with self.lock:
            parts = path.split('.')
            obj = self._config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return False
            final = parts[-1]
            if not hasattr(obj, final):
                return False
            old = getattr(obj, final)
            setattr(obj, final, value)
            self._config.modifiedby = modifiedby
            self.save()
            self.notify_listeners(path, old, value)
            return True

    def updatesection(self, section: str, values: dict, modifiedby: str = 'system') -> bool:
        with self.lock:
            if not hasattr(self._config, section):
                return False
            sectionobj = getattr(self._config, section)
            for k, v in values.items():
                if hasattr(sectionobj, k):
                    old = getattr(sectionobj, k)
                    setattr(sectionobj, k, v)
                    self.notify_listeners(f"{section}.{k}", old, v)
            self._config.modifiedby = modifiedby
            self.save()
            return True

    def addlistener(self, cb: Callable[[str, Any, Any], None]):
        self.listeners.append(cb)

    def removelistener(self, cb: Callable[[str, Any, Any], None]):
        if cb in self.listeners:
            self.listeners.remove(cb)

    def notify_listeners(self, path: str, old: Any, new: Any):
        for l in list(self.listeners):
            try:
                l(path, old, new)
            except Exception:
                pass

    def exportconfig(self) -> dict:
        return asdict(self._config)

    def importconfig(self, data: dict, modifiedby: str = 'import') -> bool:
        try:
            with self.lock:
                self._config = SystemConfig(
                    llm=LLMConfig(**data.get('llm', {})),
                    rag=RAGConfig(**data.get('rag', {})),
                    ui=UIConfig(**data.get('ui', {})),
                    security=SecurityConfig(**data.get('security', {})),
                    version=data.get('version', '4.0.0'),
                    modifiedby=modifiedby
                )
                self.save()
            return True
        except Exception:
            return False

    def resettodefaults(self, section: Optional[str] = None):
        with self.lock:
            if section is None:
                self._config = SystemConfig()
            elif section == 'llm':
                self._config.llm = LLMConfig()
            elif section == 'rag':
                self._config.rag = RAGConfig()
            elif section == 'ui':
                self._config.ui = UIConfig()
            elif section == 'security':
                self._config.security = SecurityConfig()
            self.save()


configmanager = ConfigManager()
