"""module reads the configuration: hyperparameters and general settings \
    uses .env file for general settings and tune_config.yaml for hyperparameters"""

from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, BaseSettings


class Effnet2(BaseModel):

    lr: Union[float, List[float]] = 0.0003
    dropout: Union[float, List[float]] = 0.65
    size: Union[str, List[str]] = "small"
    label_smoothing: Union[float, List[float]] = 0.0


class HyperConfig(BaseModel):
    model_arch: str  # required
    effnet2: Effnet2 = None


class GeneralConfig(BaseSettings):

    TFHUB_CACHE_DIR: str
    MODE: str = "debug"

    class Config:
        case_sensitive = False
        env_file = ".env"  # This is the key factor
        env_file_encoding = "utf-8"


def read_config():

    with open("tune_config.yaml", "r", encoding="utf-8") as file:
        hyper_config = HyperConfig(**yaml.safe_load(file)).dict()

    hyper_config_model = hyper_config[hyper_config["model_arch"]]
    hyper_config_model["model_arch"] = hyper_config["model_arch"]

    general_config = GeneralConfig().dict()

    return {**hyper_config_model, **general_config}


if __name__ == "__main__":
    config = GeneralConfig()
    print(config.dict())

    print(read_config().dict())
