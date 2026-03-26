#!/usr/bin/env python3
"""
Benchmark script to compute tokenizer throughput (bytes/second).
"""

import json
import os
import time
from pathlib import Path

from cs336_basics import BPETokenizer


def gpt2_bytes_to_unicode():
    """
    Returns a mapping between every possible byte to a printable unicode string.
    This function is taken from the GPT-2 code.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def load_tokenizer(vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None) -> BPETokenizer:
    """
    Load a BPE tokenizer from vocab and merges files.
    Similar to the load function in test_tokenizer.py
    """
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    # Load vocabulary
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    
    # Load merges
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    
    # Convert GPT-2 vocab to byte representation
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    
    # Add special tokens if provided
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token
    
    # Convert merges to byte representation
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    
    return BPETokenizer(vocab, merges, special_tokens)


def benchmark_tokenizer_encode(input_file: str | os.PathLike, tokenizer: BPETokenizer) -> dict:
    """
    Benchmark the tokenizer using encode() method.
    
    Args:
        input_file: Path to the input file
        tokenizer: BPETokenizer instance
    
    Returns:
        Dictionary with benchmark results
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_size_bytes = len(content.encode('utf-8'))
    
    # Warm up
    _ = tokenizer.encode(content[:1000] if len(content) > 1000 else content)
    
    # Benchmark
    start_time = time.perf_counter()
    token_ids = tokenizer.encode(content)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    throughput_bytes_per_sec = file_size_bytes / elapsed_time
    throughput_tokens_per_sec = len(token_ids) / elapsed_time
    
    return {
        'input_file': str(input_file),
        'file_size_bytes': file_size_bytes,
        'num_tokens': len(token_ids),
        'elapsed_time_seconds': elapsed_time,
        'throughput_bytes_per_sec': throughput_bytes_per_sec,
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'avg_bytes_per_token': file_size_bytes / len(token_ids) if token_ids else 0,
    }


def benchmark_tokenizer_encode_iterable(input_file: str | os.PathLike, tokenizer: BPETokenizer) -> dict:
    """
    Benchmark the tokenizer using encode_iterable() method.
    
    Args:
        input_file: Path to the input file
        tokenizer: BPETokenizer instance
    
    Returns:
        Dictionary with benchmark results
    """
    # Get file size
    file_size_bytes = os.path.getsize(input_file)
    
    # Warm up with a small chunk
    with open(input_file, 'r', encoding='utf-8') as f:
        first_chunk = f.readline()
        _ = list(tokenizer.encode_iterable([first_chunk]))
    
    # Benchmark with encode_iterable
    start_time = time.perf_counter()
    num_tokens = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for token_id in tokenizer.encode_iterable(f):
            num_tokens += 1
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    throughput_bytes_per_sec = file_size_bytes / elapsed_time
    throughput_tokens_per_sec = num_tokens / elapsed_time
    
    return {
        'input_file': str(input_file),
        'file_size_bytes': file_size_bytes,
        'num_tokens': num_tokens,
        'elapsed_time_seconds': elapsed_time,
        'throughput_bytes_per_sec': throughput_bytes_per_sec,
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'avg_bytes_per_token': file_size_bytes / num_tokens if num_tokens else 0,
    }


def main():
    # Define paths
    fixtures_path = Path(__file__).parent / "tests" / "fixtures"
    vocab_path = fixtures_path / "gpt2_vocab.json"
    merges_path = fixtures_path / "gpt2_merges.txt"
    input_file = fixtures_path / "tinystories_sample_5M.txt"
    
    # Verify files exist
    if not vocab_path.exists():
        print(f"Error: Vocab file not found at {vocab_path}")
        return
    if not merges_path.exists():
        print(f"Error: Merges file not found at {merges_path}")
        return
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    print("Tokenizer loaded successfully!")
    
    # Benchmark encode() method
    print(f"\nBenchmarking tokenizer.encode() on {input_file.name}...")
    results_encode = benchmark_tokenizer_encode(input_file, tokenizer)
    
    # Benchmark encode_iterable() method
    print(f"Benchmarking tokenizer.encode_iterable() on {input_file.name}...")
    results_iterable = benchmark_tokenizer_encode_iterable(input_file, tokenizer)
    
    # Display results
    print("\n" + "="*80)
    print("TOKENIZER THROUGHPUT BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nInput file: {results_encode['input_file']}")
    print(f"File size: {results_encode['file_size_bytes']:,} bytes ({results_encode['file_size_bytes'] / (1024**2):.2f} MB)")
    
    print("\n" + "-"*80)
    print("METHOD 1: tokenizer.encode() (loads entire file into memory)")
    print("-"*80)
    print(f"Number of tokens: {results_encode['num_tokens']:,}")
    print(f"Elapsed time: {results_encode['elapsed_time_seconds']:.4f} seconds")
    print(f"Throughput (bytes/second): {results_encode['throughput_bytes_per_sec']:,.2f} bytes/sec")
    print(f"Throughput (MB/second): {results_encode['throughput_bytes_per_sec'] / (1024**2):.2f} MB/sec")
    print(f"Throughput (tokens/second): {results_encode['throughput_tokens_per_sec']:,.2f} tokens/sec")
    print(f"Average bytes per token: {results_encode['avg_bytes_per_token']:.2f}")
    
    print("\n" + "-"*80)
    print("METHOD 2: tokenizer.encode_iterable() (memory-efficient streaming)")
    print("-"*80)
    print(f"Number of tokens: {results_iterable['num_tokens']:,}")
    print(f"Elapsed time: {results_iterable['elapsed_time_seconds']:.4f} seconds")
    print(f"Throughput (bytes/second): {results_iterable['throughput_bytes_per_sec']:,.2f} bytes/sec")
    print(f"Throughput (MB/second): {results_iterable['throughput_bytes_per_sec'] / (1024**2):.2f} MB/sec")
    print(f"Throughput (tokens/second): {results_iterable['throughput_tokens_per_sec']:,.2f} tokens/sec")
    print(f"Average bytes per token: {results_iterable['avg_bytes_per_token']:.2f}")
    
    # Calculate difference
    time_diff = results_encode['elapsed_time_seconds'] - results_iterable['elapsed_time_seconds']
    time_diff_pct = (time_diff / results_encode['elapsed_time_seconds']) * 100 if results_encode['elapsed_time_seconds'] > 0 else 0
    throughput_diff = results_iterable['throughput_bytes_per_sec'] - results_encode['throughput_bytes_per_sec']
    throughput_diff_pct = (throughput_diff / results_encode['throughput_bytes_per_sec']) * 100 if results_encode['throughput_bytes_per_sec'] > 0 else 0
    
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    print(f"Time difference: {time_diff:.4f} seconds ({time_diff_pct:+.2f}%)")
    print(f"Throughput difference: {throughput_diff:,.2f} bytes/sec ({throughput_diff_pct:+.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
