from trl import setup_chat_format
from unsloth import FastLanguageModel


def build_unsloth_model(
    model_name: str,
    max_seq_length: int,
    gradient_checkpointing: bool,
) -> FastLanguageModel:
    """Download a model from HF and adds the qlora adapters.

    Args:
        model_name (str): HF model name
        max_seq_length (int): max sequence length
        gradient_checkpointing (bool): Use gradient checkpointing to keep GPU ressource usage low

    Returns:
        FastLanguageModel: A tuple containing the build model and tokenizer
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        fix_tokenizer=False,
    )

    if not tokenizer.chat_template:
        model, tokenizer = setup_chat_format(model, tokenizer)
    print(f"\nChat Template: {tokenizer.chat_template}\n")

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer
