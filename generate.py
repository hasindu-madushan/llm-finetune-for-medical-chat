import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading


def generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int=128, max_len: int=512) -> str:
    """ Generate the out of the model for given text prompt """
    sample = tokenizer(prompt, truncation=True, padding=False, max_length=max_len, return_attention_mask=True)
    input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
    attention_mask = torch.tensor([sample["attention_mask"]]).to(model.device)
    
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False
    )
    
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts[0]


def stream_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    skip_special_tokens: bool = True
):
    """ Generate the output of the model as a text stream for given text prompt 
    
    Args:
        do_sample: to enable the sampling for the generation
    """
    device = model.device
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Initialize streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=skip_special_tokens)

    # Generation arguments
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    # Run generation in a separate thread to allow streaming
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # Stream tokens as they are generated
    for token in streamer:
        yield token
