import json



class BPETokenizerParams:
    vocab: dict[int, bytes] #int -> bytes
    merges: dict[tuple[int, int], int] #(token1, token2) -> int

def save_bpe(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        filename_prefix: str
):
    vocab_path = f"{filename_prefix}_vocab.json"
    json_vocab = {
        str(idx): token_bytes.decode('latin-1')
        for idx, token_bytes in vocab.items()
    }

    with open(vocab_path, "w", encoding='utf-8') as f:
        json.dump(json_vocab, f, indent=2, ensure_ascii=True)

    merges_path = f"{filename_prefix}_merges.txt"
    with open(merges_path, "w", encoding='utf-8') as f:
        f.write("#version: 0.2\n")

        for p1, p2 in merges:
            s1 = p1.decode('latin-1')
            s2 = p2.decode('latin-1')
            f.write(f"{s1} {s2}\n")

    print(f"Save vocab to {vocab_path}")
    print(f"Save merges to {merges_path}")

def load_bpe(vocab_path, merges_path):
    with open(vocab_path, "r", encoding='utf-8') as f:
        json_vocab = json.load(f)

    vocab = {
        int(idx): token_str.encode('latin-1')
        for idx , token_str in json_vocab.items()
    }
    merges = []
    with open(merges_path, "r", encoding='utf-8') as f:
        start_line = 1 if f.readline().startswith("#") else 0
        f.seek(0)
        lines = f.readlines()[start_line:]

    for line in lines:
        parts = line.split(" ")
        if len(parts) != 2: continue
        p1 = parts[0].encode('latin-1')
        p2 = parts[1].strip().encode('latin-1')
        merges.append((p1, p2))

    return vocab, merges