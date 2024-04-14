from pathlib import Path

import fire
from beam import App, Image, Runtime, Volume, VolumeType
from dotenv import find_dotenv, load_dotenv

from training_pipeline.api.train import TrainAPI
from training_pipeline.config import TrainingConfig

training_app = App(
    name="train_qa",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(path="./qa_dataset", name="qa_dataset"),
        Volume(
            path="./output",
            name="train_qa_output",
            volume_type=VolumeType.Persistent,
        ),
        Volume(path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent),
    ],
)


@training_app.run()
def train(
    config_file: str,
    output_dir: str,
):
    """
    Trains a machine learning model using the specified configuration file and dataset.

    Args:
        config_file (str): The path to the configuration file for the training process.
        output_dir (str): The directory where the trained model will be saved.
        dataset_dir (str): The directory where the training dataset is located.
        env_file_path (str, optional): The path to the environment variables file. Defaults to ".env".
        logging_config_path (str, optional): The path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str, optional): The directory where the trained model will be cached. Defaults to None.
    """
    load_dotenv(find_dotenv())
    config_file = Path(config_file)
    output_dir = Path(output_dir)

    training_config = TrainingConfig.from_yaml(config_file, output_dir)
    training_api = TrainAPI.from_config(
        config=training_config,
    )
    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
