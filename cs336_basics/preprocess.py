import os
import json
import numpy as np
from cs336_basics.tokenizer import BPETokenizer
from tqdm import tqdm
import multiprocessing as mp

# Globals for the worker processes
global_tokenizer = None

def init_worker(vocab, merges, special_tokens):
    global global_tokenizer
    global_tokenizer = BPETokenizer(vocab, merges, special_tokens)

def worker_encode_batch(docs):
    results = []
    for doc in docs:
        tokens = global_tokenizer.encode(doc)
        bytes_len = len(doc.encode('utf-8'))
        results.append((tokens, bytes_len))
    return results

def doc_batches_iterator(input_path, chunk_size, eot_token, batch_size=500):
    with open(input_path, "r", encoding="utf-8") as f:
        remainder = ""
        batch = []
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            content = remainder + chunk
            docs = content.split(eot_token)
            
            remainder = docs.pop()
            
            for doc in docs:
                doc_str = doc.strip()
                if doc_str:
                    batch.append(doc_str)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
        
        last_doc = remainder.strip()
        if last_doc:
            batch.append(last_doc)
            
        if batch:
            yield batch

def process_to_shards(input_path: str, output_dir: str, tokenizer: BPETokenizer, shard_size_tokens=1e9):
    os.makedirs(output_dir, exist_ok=True)

    file_size = os.path.getsize(input_path)
    chunk_size = 1024 * 1024 * 1024  # 1GB chunks

    eot_token = "<|endoftext|>"
    eot_id = tokenizer.convert_tokens_to_ids([eot_token])[0]

    shard_idx = 0
    all_shard_metadata = []

    token_buffer = []    # flat list of tokens
    lengths_buffer = []  # length of each document

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
    
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Reading")

    vocab = tokenizer.vocab
    merges = tokenizer.merges
    special_tokens = tokenizer.special_tokens

    # Set up pool
    pool = mp.Pool(
        processes=max(1, mp.cpu_count() - 1),
        initializer=init_worker,
        initargs=(vocab, merges, special_tokens)
    )

    total_docs = 0
    total_tokens = 0

    batch_iterator = doc_batches_iterator(input_path, chunk_size, eot_token, batch_size=2000)
    eot_bytes_len = len(eot_token.encode('utf-8'))
    
    for batch_results in pool.imap(worker_encode_batch, batch_iterator):
        for tokens, doc_bytes in batch_results:
            pbar.update(doc_bytes + eot_bytes_len)
            
            tokens.append(eot_id)
            total_docs += 1
            total_tokens += len(tokens)
            token_buffer.extend(tokens)
            lengths_buffer.append(len(tokens))
            pbar.set_postfix(docs=total_docs, tokens=f"{total_tokens:,}", shards=shard_idx)

            if len(token_buffer) >= shard_size_tokens:
                meta = save_current_shard(shard_idx, token_buffer, lengths_buffer)
                all_shard_metadata.append(meta)
                num_tokens = meta["tokens"]
                tqdm.write(f"  Saved shard {shard_idx:04d} ({num_tokens:,} tokens)")
                shard_idx += 1
                token_buffer, lengths_buffer = [], []

    pool.close()
    pool.join()
    pbar.close()

    if token_buffer:
        meta = save_current_shard(shard_idx, token_buffer, lengths_buffer)
        all_shard_metadata.append(meta)

    print(f"\nPreprocessing complete: {total_docs:,} docs, {total_tokens:,} tokens, {len(all_shard_metadata)} shards")

    manifest = {
        "tokenizer": tokenizer.__class__.__name__,
        "vocab_size": len(vocab),
        "dtype": "uint16",
        "total_tokens": sum(m["tokens"] for m in all_shard_metadata),
        "shards": all_shard_metadata
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    tokenizer = BPETokenizer.from_files("data/bpe_vocab.json", "data/bpe_merges.txt", ["<|endoftext|>"])
    process_to_shards("data/TinyStoriesV2-GPT4-valid.txt", "data/valid-bin", tokenizer)
    print("Done!")
