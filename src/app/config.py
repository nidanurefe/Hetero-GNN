from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import List


@dataclass
class RawDatasetConfig:
    name: str
    path: str


@dataclass
class RawConfig:
    target: RawDatasetConfig
    sources: List[RawDatasetConfig]


@dataclass
class DataConfig:
    processed_dir: str
    raw: RawConfig


@dataclass
class ModelConfig:
    type: str = "gat"
    hidden_channels: int = 32
    out_channels: int = 32
    num_layers: int = 2
    heads: int = 2
    dropout: float = 0.1
    edge_dim: int = 4


@dataclass
class TrainingConfig:
    loss_type: str = "bpr_softplus"
    num_epochs: int = 25
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
_config_cache: Config | None = None

def load_config() -> Config:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    with _DEFAULT_CONFIG_PATH.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    data_cfg = raw_cfg["data"]
    train_cfg = raw_cfg["training"]
    model_cfg = raw_cfg["model"]

    target_cfg = data_cfg["raw"]["target"]
    sources_cfg = data_cfg["raw"]["sources"]

    logging_cfg = raw_cfg.get("logging", {})

    cfg = Config(
        data=DataConfig(
            processed_dir=data_cfg["processed_dir"],
            raw=RawConfig(
                target=RawDatasetConfig(**target_cfg),
                sources=[RawDatasetConfig(**s) for s in sources_cfg],
                
            ),
        ),
        training=TrainingConfig(**train_cfg),
        model=ModelConfig(**model_cfg),
        logging=LoggingConfig(**logging_cfg),
    )

    _config_cache = cfg
    return cfg

def get_config() -> Config:
    return load_config()