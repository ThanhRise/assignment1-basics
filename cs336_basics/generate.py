import torch
from cs336_basics.model import TransformerLM
from cs336_basics.attention import softMax

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompts: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None) -> torch.Tensor:
    """
    Generate the next token in the sequence for each prompt in the batch.
    
    Args:
        model (torch.nn.Module): The model to use for generation.
        prompts (torch.Tensor): The prompts to use for generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_token_id (int): The end of sequence token id.
        temperature (float): The temperature to use for generation.
        top_k (int | None): The top k to use for generation.
        top_p (float | None): The top p to use for generation.
    Returns:
        torch.Tensor: The generated tokens.
    """
    model.eval()
    batch_size, seq_len = prompts.shape
    
    for _ in range(max_new_tokens):
        logits = model(prompts)
        next_token_logits = logits[:, -1, :] / temperature
        
        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            if top_k is not None:
                top_k = min(max(top_k, 1), next_token_logits.size(-1))
                top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                kth_value = top_k_values[:, -1, None]
                indices_to_remove = next_token_logits < kth_value
                next_token_logits[indices_to_remove] = float('-inf')
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = softMax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            probs = softMax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        prompts = torch.cat([prompts, next_token], dim=-1)
        if (next_token == eos_token_id).any():
            break
    return prompts

if __name__ == "__main__":
    import argparse
    from cs336_basics.tokenizer import BPETokenizer
    
    parser = argparse.ArgumentParser(description="Generate text using TransformerLM")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing one prompt per line")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--bpe_vocab", type=str, default="data/bpe_vocab.json", help="Path to BPE vocab")
    parser.add_argument("--bpe_merges", type=str, default="data/bpe_merges.txt", help="Path to BPE merges")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-P (nucleus) sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (e.g. cpu or cuda)")
    
    args = parser.parse_args()
    
    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(args.bpe_vocab, args.bpe_merges, special_tokens=["<|endoftext|>"])
    # If using the same vocabulary dimension setup as training:
    eos_token_id = tokenizer.vocab["<|endoftext|>"]
    
    # 2. Encode Prompts and Batch via Left-Padding
    if args.prompt is not None:
        lines = [args.prompt]
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide either --prompt or --prompt_file")

    encoded_prompts = [tokenizer.encode(line) for line in lines]
    max_len = max(len(enc) for enc in encoded_prompts)
    
    # Left pad with eos_token_id to align the final token for causal sequence prediction
    padded_prompts = []
    for enc in encoded_prompts:
        pad_len = max_len - len(enc)
        padded_prompts.append([eos_token_id] * pad_len + enc)
        
    input_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=args.device)
    
    # 3. Initialize Model
    print("Loading model...")
    # NOTE: Ensure these architecture parameters match what your checkpoint was trained with
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab), 
        context_length=1024, 
        num_layers=8, 
        num_heads=12, 
        d_model=768, 
        d_ff=2048, 
        rope_theta=10000
    )
    
    # Safely load the weights
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    
    # Check if this is a dict containing the "model" key (like our trainer saves)
    if "model" in checkpoint:
        # Strip out any DDP "module." prefixes if the checkpoint was saved under DDP incorrectly
        state_dict = checkpoint["model"]
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        model.load_state_dict(checkpoint)
        
    model.to(args.device)
    
    # 4. Generate Output
    print(f"\nGenerating {len(lines)} sequences in a single batch...")
    output_tensor = generate(
        model=model,
        prompts=input_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
        
    # 5. Decode and Print
    output_ids = output_tensor.tolist()
    
    print("\n" + "="*30 + " GENERATED TEXT " + "="*30)
    for i, seq_ids in enumerate(output_ids):
        # Strip the padding tokens from the left side before decoding
        pad_len = max_len - len(encoded_prompts[i])
        actual_output = seq_ids[pad_len:]
        final_text = tokenizer.decode(actual_output)
        
        print(f"\n--- Sequence {i+1} ---")
        print(final_text)
    print("\n" + "="*76 + "\n")