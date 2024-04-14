import os
from pathlib import Path
from typing import Any, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel

import wandb
from training_pipeline import static
from training_pipeline.config import TrainingConfig
from training_pipeline.model import build_unsloth_model
from training_pipeline.prompt_formatting import prompt_formatting_instruct


class TrainAPI:
    """
    Creates a training class for training a model using unsloth.

    Args:
        model_id (str): A Huggingface model id.
        training_arguments (TrainingArguments): the training argumetnts for SFTTrainer
        dataset_path (str): A HuggingFace face dataset path
        name (str, optional): The name of the training API. Defaults to "training-api"
        max_seq_length (int, optional): The maximum sequence length. Defaults to 1024.
        model_cache_dir (Path, optional): The directory to cache the model. Defaults to static.CACHE_DIR
    """

    def __init__(
        self,
        model_id: str,
        training_arguments: TrainingArguments,
        dataset_path: str,
        name: str = "training-api",
        max_seq_length: int = 1024,
        gradient_checkpointing: bool = True,
        model_cache_dir: Path = static.CACHE_DIR,
    ):
        self._model_id = model_id
        self._training_arguments = training_arguments
        self._name = name
        self._max_seq_length = max_seq_length
        self._gradient_checkpointing = gradient_checkpointing
        self._model_cache_dir = model_cache_dir

        self._training_dataset, self._validation_dataset = self.load_data(dataset_path)
        self._model, self._tokenizer = self.load_model()

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
    ):
        """
        Creates a TrainingAPI instance from a TrainingConfig object.

        Args:
            config (TrainingConfig): The training configuration.

        Returns:
            TrainingAPI: A TrainingAPI instance.
        """

        return cls(
            model_id=config.model["hf_repo"],
            training_arguments=config.training,
            max_seq_length=config.model["max_seq_length"],
            gradient_checkpointing=config.model["gradient_checkpointing"],
            dataset_path=config.dataset["dataset_path"],
        )

    def load_model(
        self,
    ) -> Tuple[Any, Any | PreTrainedTokenizer | PreTrainedTokenizerFast]:
        """
        Loads the model.

        Returns:
            Tuple[Any, Any | PreTrainedTokenizer | PreTrainedTokenizerFast]: A tuple containing the model and  tokenizer
        """

        print(f"Loading model {self._model_id}")

        model, tokenizer = build_unsloth_model(
            model_name=self._model_id,
            max_seq_length=self._max_seq_length,
            gradient_checkpointing=self._gradient_checkpointing,
        )

        return model, tokenizer

    def load_data(self, dataset_path: str) -> Tuple[Dataset, Dataset]:
        """
        Loads the training and validation datasets.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        """

        print(f"Loading QA datasets from HF")

        dataset = load_dataset(dataset_path, split="train")
        dataset = dataset.shuffle()  # .select(range(200))  #! Test
        print(dataset)
        dataset = dataset.map(
            prompt_formatting_instruct,
            remove_columns=dataset.features,
            batched=False,
        )
        dataset = dataset.train_test_split(test_size=0.01)
        training_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        print(f"Training dataset size: {len(training_dataset)}")
        print(f"Validation dataset size: {len(eval_dataset)}")

        return training_dataset, eval_dataset

    def train(self) -> SFTTrainer:
        """
        Trains the model.

        Returns:
            SFTTrainer: The trained model.
        """

        print("Initializing WandB...")

        wandb.login(key=os.getenv("WANDB"))
        wandb.init()
        os.environ["WANDB_PROJECT"] = "ASN_RAG"
        os.environ["WANDB_LOG_MODEL"] = "end"

        print("Training model...")

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=self._training_dataset,
            max_seq_length=self._max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=self._training_arguments,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
        )
        trainer.train()
        self._evaluation(trainer.model)
        wandb.finish()
        return trainer

    def _evaluate_sample(self, sample: dict, model: FastLanguageModel) -> str:
        """evaluates on one example

        Args:
            sample (dict): sample from eval set
            model (FastLanguageModel): Model to evaluate

        Returns:
            (str): the predicted answer
        """
        prompt = self._tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([prompt], return_tensors="pt", padding=False).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        outputs = self._tokenizer.batch_decode(outputs)
        predicted_answer = outputs[0][(len(prompt) + 7) :].strip()
        return predicted_answer

    def _evaluation(self, _model: FastLanguageModel) -> None:
        """Evaluates the model on the validation set

        Args:
            _model (FastLanguageModel): Model for inference

        Returns:
            (None): Accuracy to stdout
        """
        FastLanguageModel.for_inference(_model)

        answers = wandb.Table(columns=["id", "pred"])
        number_of_eval_samples = len(self._validation_dataset)

        # iterate over eval dataset and predict
        for s in tqdm(self._validation_dataset.shuffle().select(range(number_of_eval_samples))):
            pred = self._evaluate_sample(s, _model)
            answers.add_data(s["messages"][0]["content"], pred)

        wandb.log({"answers": answers})

        return print(f"Evaluation done. See results in WandB.")
