import json
from typing import Iterable
import regex as re
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

class BPETokenizer:
    rankMap = None
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        BPETokenizer.rankMap = {pair: rank for rank, pair in enumerate(self.merges)}

    @staticmethod
    def bytesToTokens(text: str) -> list[bytes]:
        return [
            bytes([b])
            for m in re.finditer(PAT, text)
            for b in m.group().encode('utf-8')
        ]
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab, merges = load_bpe(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def initializeStructures(indices: list[bytes]) -> tuple[list[int], list[int], list[bytes | None], list[tuple[int, int]]]:
        if BPETokenizer.rankMap == None:
            raise AttributeError
        input_length = len(indices)
        nextP = [-1] * input_length
        prevP = [-1] * input_length
        valueP = [None] * input_length
        heapRank = []

        for i in range(input_length):
            valueP[i] = indices[i]
            prevP[i] = i - 1
            nextP[i] = i + 1
            if i < input_length - 1:
                pair = (indices[i], indices[i+1])
                rank =  BPETokenizer.rankMap.get(pair, float('inf'))
                if rank != float('inf'):
                    heapq.heappush(heapRank, (rank, i))

        nextP[input_length - 1] = -1
        return (nextP, prevP, valueP, heapRank)

    def BPEInference(
        self,
        nextP: list[int],
        prevP: list[int],
        valueP: list[bytes | None],
        heapRank: list[tuple[int, int]]
    ) -> list[int]: 
        
        while len(heapRank) != 0:
            rank, idx = heapRank.pop()

            if valueP[idx] == None: continue
            next_idx = nextP[idx]
            if next_idx == -1: continue
            current_pair = (valueP[idx], valueP[next_idx])
            current_rank = BPETokenizer.rankMap.get(current_pair, float('inf'))
            if current_rank != rank: continue
            
            new_token = self.merges[current_pair]
            valueP[idx] = new_token

            node_to_remove = next_idx
            valueP[node_to_remove] = None
            new_next = nextP[node_to_remove]
            nextP[idx] = new_next

            if new_next != -1: prevP[new_next] = idx

            prev_idx = prevP[idx]
            if prev_idx != -1:
                pair = (valueP[prev_idx], valueP[idx])
                rank = BPETokenizer.rankMap.get(pair, float('inf'))
                if rank != float('inf'):
                    heapq.heappush(heapRank, (rank, prev_idx))

            if new_next != -1:
                pair = (valueP[idx], valueP[new_next])
                rank = BPETokenizer.rankMap.get(pair, float('inf'))
                if rank != float('int'):
                    heapq.heappush(heapRank, (rank, idx))

        results = []
        curr = 0
        while curr != -1 :
            results.append(valueP[curr])
            curr = nextP[curr]
        
        return results

    def encode(self, text: str) -> list[int]:
        indices = self.bytesToTokens(text)
        nextP, prevP, valueP, heapRank = self.initializeStructures(indices, self.merges)
        results = self.BPEInference(nextP, prevP, valueP, heapRank)
        return results


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        bytes_list = list(map(self.vocab, ids))
        return b"".join(bytes_list).decode('utf-8')

   
            