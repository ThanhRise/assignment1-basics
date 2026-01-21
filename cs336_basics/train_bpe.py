import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex as re
from collections import Counter, defaultdict
from functools import reduce

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def run_train_bpe_backend(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pre_token_count = count_pre_token(input_path, special_tokens)
    vocab_list = []
    byte_start_offset = len(special_tokens)

    current_vocab = {i: s.encode("utf-8") for i, s in enumerate(special_tokens)}
    current_vocab = current_vocab | {i + byte_start_offset: bytes([i]) for i in range(256)}

    for byte_seq, freq in pre_token_count.items():
        vocab_list.append({
            'tokens': [bytes([b]) for b in byte_seq],
            'freq': freq
        })

    next_token_ids = 256 + len(special_tokens)
    merges = []

    stats = defaultdict(int)
    indices = defaultdict(dict)

    for i, word_obj in enumerate(vocab_list):
        tokens = word_obj['tokens']
        freq = word_obj['freq']
        for j in range(len(tokens) -1):
            pair = (tokens[j], tokens[j+1])
            stats[pair] += freq
            indices[pair][i] = 1

    current_vocab_size = len(current_vocab)

    while current_vocab_size < vocab_size:
        if not stats:
            break

        best_pair = max(stats, key=lambda p: (stats[p], p[0], p[1]))
        if stats[best_pair] < 1:
            break

        #Record Merge
        merges.append(best_pair)
        new_token_bytes = best_pair[0] + best_pair[1]
        current_vocab[next_token_ids] = new_token_bytes
        word_to_update = list(indices[best_pair].keys())

        for word_idx in word_to_update:
            word_obj = vocab_list[word_idx]
            tokens = word_obj['tokens']
            freq = word_obj['freq']

            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    if i > 0:
                        prev_pair = (tokens[i-1], tokens[i])
                        stats[prev_pair] -= freq
                        if  stats[prev_pair] == 0: del  stats[prev_pair] 

                    if i < len(tokens) -2:
                        next_pair = (tokens[i+1], tokens[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair]==0: del stats[next_pair]

                    tokens[i] =  current_vocab[next_token_ids]
                    del tokens[i+1]

                    if i > 0:
                        new_prev_pair = (tokens[i-1], tokens[i])
                        stats[new_prev_pair] += freq
                        indices[new_prev_pair][word_idx] = 1

                    if i < len(tokens) - 1:
                        new_next_pair = (tokens[i], tokens[i+1])
                        stats[new_next_pair] += freq
                        indices[new_next_pair][word_idx] = 1

                else: i+=1
        del stats[best_pair]
        del indices[best_pair]
        next_token_ids+=1
        current_vocab_size+=1

    return current_vocab, merges
    
        
def count_pre_token(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter:
    file_size = os.path.getsize(input_path)
    with open(input_path, "rb") as f:
        num_chunk = max(1, int(file_size / 50000000))
        special_tokens_bytpe = [s.encode() for s in special_tokens]
        boundaries = find_chunk_boundaries(f, num_chunk, special_tokens_bytpe)
        results = Counter()
        num_processes = min(16, num_chunk)

        with mp.Pool(processes=num_processes) as pool:
            # Buffer to hold arguments for one batch
            batch_args = []
            
            # Iterate through boundaries pairs (start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_text = f.read(end - start).decode("utf-8", errors="ignore")

                batch_args.append((chunk_text, special_tokens))
                
                if len(batch_args) == num_processes:
                    # print(f"Processing batch of {len(batch_args)} chunks...")
                    batch_results = pool.imap_unordered(pre_tokenization, batch_args)
                    
                    for res in batch_results:
                        results.update(res)

                    batch_args = []

            # 4. CRITICAL: Process the remainder (leftover chunks)
            if batch_args:
                # print(f"Processing final tail of {len(batch_args)} chunks...")
                batch_results = pool.imap_unordered(pre_tokenization, batch_args)
                for res in batch_results:
                    results.update(res)
    return results


def pre_tokenization(
    args  
) -> Counter    :
    chunk , special_tokens = args
    if not special_tokens:
        return
    
    # Escape special tokens
    escape_tokens = [re.escape(tok) for tok in special_tokens]
    pattern = "|".join(escape_tokens)
    segments = re.split(pattern, chunk)

    segments = [s for s in segments if s]

    counts = Counter()  
    for seg in segments:
        matches = re.finditer(PAT, seg)
        for match in matches:
            token_str = match.group().encode('utf-8')
            counts[token_str] += 1

    return counts


if __name__ == "__main__":
    run_train_bpe_backend("data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])