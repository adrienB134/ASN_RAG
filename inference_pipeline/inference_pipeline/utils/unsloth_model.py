# from peft import PeftConfig, PeftModel
# from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from typing import Literal, Optional

from dsp.modules.lm import LM

# from dsp.modules.finetuning.finetune_hf import preprocess_prompt


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class UnslothModel(LM):
    def __init__(
        self,
        model: str,
        checkpoint: Optional[str] = None,
        is_client: bool = False,
        hf_device_map: Literal[
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ] = "auto",
        token: Optional[str] = None,
        use_chat_template: bool = False,
    ):
        """wrapper for Hugging Face models using unsloth for faster inference

        Args:
            model (str): HF model identifier to load and use
            checkpoint (str, optional): load specific checkpoints of the model. Defaults to None.
            is_client (bool, optional): whether to access models via client. Defaults to False.
            hf_device_map (str, optional): HF config strategy to load the model.
                Recommended to use "auto", which will help loading large models using accelerate. Defaults to "auto".
        """

        super().__init__(model)
        self.provider = "hf"
        self.is_client = is_client
        self.device_map = hf_device_map

        hf_autoconfig_kwargs = dict(token=token or os.environ.get("HF_TOKEN"))
        hf_autotokenizer_kwargs = hf_autoconfig_kwargs.copy()
        hf_automodel_kwargs = hf_autoconfig_kwargs.copy()
        if not self.is_client:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from unsloth import FastLanguageModel
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "You need to install unsloth library to use fast HF models.",
                ) from exc
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(model)
                FastLanguageModel.for_inference(self.model)
                self.rationale = True

                self.drop_prompt_from_output = False
            except ValueError as e:
                print(e)
                print("Falling back to Transformers")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model if checkpoint is None else checkpoint,
                    device_map=self.device_map,
                    **hf_automodel_kwargs,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model,
                    **hf_autotokenizer_kwargs,
                )
                self.drop_prompt_from_output = True
        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        # TODO: Add caching
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        # print(prompt)
        if isinstance(prompt, dict):
            try:
                prompt = prompt["messages"][0]["content"]
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # print(kwargs)
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # ,, eos_token_id=self.tokenizer.pad_token_id
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [{"text": c} for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]
