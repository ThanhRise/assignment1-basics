



class BPETokenizerParams:
    vocab: dict[int, bytes] #int -> bytes
    merges: dict[tuple[int, int], int] #(token1, token2) -> int

