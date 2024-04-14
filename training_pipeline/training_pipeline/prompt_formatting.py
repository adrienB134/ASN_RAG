def prompt_formatting_chat(sample):
    # Convert dataset to OAI messages
    system_message = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    INPUT:
    {input}"""
    return {
        "messages": [
            {"role": "system", "content": system_message.format(input=sample["input"])},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def prompt_formatting_instruct(sample):
    # Convert dataset to OAI messages
    system_message = """You are CroissantLLM an helpful assistant who can speak french and english."""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}

    ### Input:
    {input}"""

    return {
        "messages": [
            # {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": alpaca_prompt.format(instruction=sample["instruction"], input=sample["input"]),
            },
            {"role": "assistant", "content": sample["output"]},
        ],
    }
