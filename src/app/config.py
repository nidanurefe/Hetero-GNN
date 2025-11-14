from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    raw_path: str
    processed_dir: str


@dataclass
class ModelConfig:
    type: str = "gat"
    hidden_channels: int = 32
    out_channels: int = 32
    num_layers: int = 2
    heads: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    loss_type: str = "bpr_softplus"
    num_epochs: int = 20
    batch_size: int = 512
    val_batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    reg_lambda: float = 1e-4
    device: str = "auto"   # "auto" | "cpu" | "cuda"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig


# Config Loader

_DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_cached_config: Config | None = None


def load_config(path: Path | str = _DEFAULT_CONFIG_PATH) -> Config:
    global _cached_config

    if _cached_config is not None:
        return _cached_config

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw["model"])
    training_cfg = TrainingConfig(**raw["training"])
    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    _cached_config = Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
    )
    return _cached_config


def get_config() -> Config:
    return load_config()