import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars


@dataclass
class ThresholdConfig:
    health_potion: float = 0.5
    mana_potion: float = 0.3
    chicken: float = 0.2
    merc_health_potion: float = 0.4


@dataclass
class TimingConfig:
    loop_delay: float = 0.05
    after_teleport: float = 0.1
    after_cast: float = 0.3
    after_pickup: float = 0.1
    town_wait: float = 0.5


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "gpt-4o"
    evaluate_items: bool = True
    max_tokens: int = 500
    cache_evaluations: bool = True


@dataclass
class Config:
    """Main configuration container."""
    window_width: int = 1280
    window_height: int = 720
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runs_enabled: list = field(default_factory=lambda: ["mephisto"])
    debug_mode: bool = False

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a raw config value by dot-notation key."""
        keys = key.split(".")
        value = self._raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return Config()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Parse into structured config
    config = Config(_raw=raw)

    # Window
    if "window" in raw:
        config.window_width = raw["window"].get("expected_width", 1280)
        config.window_height = raw["window"].get("expected_height", 720)

    # Thresholds
    if "thresholds" in raw:
        config.thresholds = ThresholdConfig(
            health_potion=raw["thresholds"].get("health_potion", 0.5),
            mana_potion=raw["thresholds"].get("mana_potion", 0.3),
            chicken=raw["thresholds"].get("chicken", 0.2),
            merc_health_potion=raw["thresholds"].get("merc_health_potion", 0.4),
        )

    # Timing
    if "timing" in raw:
        config.timing = TimingConfig(
            loop_delay=raw["timing"].get("loop_delay", 0.05),
            after_teleport=raw["timing"].get("after_teleport", 0.1),
            after_cast=raw["timing"].get("after_cast", 0.3),
            after_pickup=raw["timing"].get("after_pickup", 0.1),
            town_wait=raw["timing"].get("town_wait", 0.5),
        )

    # LLM
    if "llm" in raw:
        config.llm = LLMConfig(
            enabled=raw["llm"].get("enabled", True),
            model=raw["llm"].get("model", "gpt-4o"),
            evaluate_items=raw["llm"].get("evaluate_items", True),
            max_tokens=raw["llm"].get("max_tokens", 500),
            cache_evaluations=raw["llm"].get("cache_evaluations", True),
        )

    # Runs
    if "runs" in raw and "enabled" in raw["runs"]:
        config.runs_enabled = raw["runs"]["enabled"]

    # Debug
    if "debug" in raw:
        config.debug_mode = raw["debug"].get("verbose_logging", False)

    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
