import os
import json
import numpy as np
from cs336_basics.tokenizer import BPETokenizer
from tqdm import tqdm

def process_to_shards(input_path: str, output_dir: str, tokenizer: BPETokenizer, shard_size_tokens=10**7):
    os.makedirs(output_dir, exist_ok=True)

    eot_token = "<|endoftext|>"
    eot_id = tokenizer.convert_tokens_to_ids([eot_token])[0]

    shard_idx = 0
    all_shard_metadata = []

    token_buffer = []
    lengths_buffer = []
    remainder = ""

    def save_current_shard(s_idx, tokens, lengths):
        name = f"shard_{s_idx:04d}"
        bin_path = os.path.join(output_dir, f"{name}.bin")
        idx_path = os.path.join(output_dir, f"{name}.idx")

        token_arr = np.array(tokens, dtype=np.uint16)
        token_arr.tofile(bin_path)

        lengths_arr = np.array(lengths, dtype=np.uint32)
        offsets_arr = np.zeros(len(lengths_arr), dtype=np.uint64)
        offsets_arr[1:] = np.cumsum(lengths_arr)[:-1]

        with open(idx_path, "wb") as f:
            f.write(np.array([len(lengths_arr)], dtype=np.uint32).tobytes())
            f.write(offsets_arr.tobytes())
            f.write(lengths_arr.tobytes())

        return {"id": s_idx, "file": f"{name}.bin", "tokens": len(tokens)}
    
    with open(input_path, "r", encoding='utf-8') as f:
        while True:
            chunk = f.read(1024*1024*16)
            if not chunk:
                break

            content = remainder + chunk
            docs = content.split(eot_token)

            remainder = docs.pop()
            for doc in docs:
                if not doc.strip(): continue
                tokens = tokenizer.encode(doc.strip())
                tokens.append(eot_id)

                token_buffer.append(tokens)
                lengths_buffer.append(len(tokens))

                if len(token_buffer) > shard_size_tokens:
                    meta = save_current_shard(shard_idx, token_buffer, lengths_buffer)
                    all_shard_metadata.append(meta)
                    shard_idx +=1
                    token_buffer , lengths_buffer = [], []

    last_content = remainder.strip()    
    if last_content:
        tokens = tokenizer.encode(last_content)
        tokens.append(eot_id)
        token_buffer.append(tokens)
        lengths_buffer.append(len(tokens))

    if token_buffer:
        meta = save_current_shard(shard_idx, token_buffer, lengths_buffer)
        all_shard_metadata.append(meta)

    manifest = {
        "tokenizer" : tokenizer.__class__.__name__ ,
        "dtype": "uint16",
        "total_tokens": sum(m["tokens"] for m in all_shard_metadata),
        "shards": all_shard_metadata
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files("data/bpe_vocab.json", "data/bpe_merges.txt", ["<|endoftext|>"])
    process_to_shards("data/owt_train.txt", "data/train-bin", tokenizer)
    print("Done!")