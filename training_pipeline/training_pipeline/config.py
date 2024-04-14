from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from transformers import TrainingArguments


def load_yaml(path: Path) -> dict:
    """
    Load a YAML file from the given path and return its contents as a dictionary.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class TrainingConfig:
    """
    Training configuration class used to load and store the training configuration.

    Attributes:
    -----------
    training : TrainingArguments
        The training arguments used for training the model.
    model : Dict[str, Any]
        The dictionary containing the model configuration.
    """

    training: TrainingArguments
    model: Dict[str, Any]
    dataset: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path) -> "TrainingConfig":
        """
        Load a configuration file from the given path.

        Parameters:
        -----------
        config_path : Path
            The path to the configuration file.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingConfig
            The training configuration object.
        """

        config = load_yaml(config_path)

        config["training"] = cls._dict_to_training_arguments(training_config=config["training"], output_dir=output_dir)

        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(cls, training_config: dict, output_dir: Path) -> TrainingArguments:
        """
        Build a HF TrainingArguments object from a configuration dictionary.

        Parameters:
        -----------
        training_config : dict
            The dictionary containing the training configuration.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingArguments
            The training arguments object.
        """

        return TrainingArguments(
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            gradient_checkpointing=training_config["gradient_checkpointing"],
            warmup_ratio=training_config["warmup_ratio"],
            num_train_epochs=training_config["num_train_epochs"],
            learning_rate=float(training_config["learning_rate"]),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config["logging_steps"],
            optim=training_config["optim"],
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            seed=training_config["seed"],
            output_dir=training_config["output_dir"],
            report_to=training_config["report_to"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            evaluation_strategy=training_config["evaluation_strategy"],
            push_to_hub=training_config["push_to_hub"],
        )
