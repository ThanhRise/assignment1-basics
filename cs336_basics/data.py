from __future__ import annotations

import os
import json
import bisect

import numpy as np
import torch


class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, context_window=1024):
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.meta = json.load(f)
        self.data_dir = data_dir
        self.dtype = np.uint16 if self.meta.get("dtype") == "uint16" else np.int32
        self.context_window = context_window
        self.shard_files = []
        self.cumulative_docs = []
        self.shard_indices = []
        total_docs = 0

        for shard in self.meta["shards"]:
            shard_id = shard["id"]
            bin_path = os.path.join(self.data_dir, f"shard_{shard_id:04d}.bin")
            idx_path = os.path.join(self.data_dir, f"shard_{shard_id:04d}.idx")

            self.shard_files.append(bin_path)
            byte_per_token = 2 if self.dtype==np.uint16 else 4
            shard_token_capacity = os.path.getsize(bin_path) // byte_per_token

            with open(idx_path, 'rb') as f:
                count = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                offsets = np.frombuffer(f.read(count * 8), dtype=np.uint64)
                lengths = np.frombuffer(f.read(count * 4), dtype=np.uint32)

            valid_mask = (offsets + self.context_window + 1) <= shard_token_capacity

            valid_offsets = offsets[valid_mask]
            valid_lengths = lengths[valid_mask]
            valid_count = len(valid_offsets)

            self.shard_indices.append({
                "offsets": valid_offsets,
                "lengths": valid_lengths
            })
            total_docs+= valid_count
            self.cumulative_docs.append(total_docs)
        
        self.total_docs = total_docs
        self._memmaps = {}

    
    def __len__(self):
        return self.total_docs
    def _get_memmap(self, shard_idx):
        """Lazily loads and caches to the numpy memmap for a specific shard."""
        if shard_idx not in self._memmaps:
            bin_path = self.shard_files[shard_idx]
            self._memmaps[shard_idx] = np.memmap(bin_path, dtype=self.dtype, mode='r')
        return self._memmaps[shard_idx]
    def __getitem__(self, global_idx):
        if global_idx < 0 or global_idx >= self.total_docs:
            raise IndexError("Dataset index out of range")

        shard_idx = bisect.bisect_right(self.cumulative_docs, global_idx)
        if shard_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_docs[shard_idx - 1]
        offset = self.shard_indices[shard_idx]["offsets"][local_idx]
        length = self.shard_indices[shard_idx]["lengths"][local_idx]

        data_map = self._get_memmap(shard_idx)
        sample_np = data_map[offset : offset+self.context_window+1]
        sample_tensor = torch.from_numpy(sample_np.astype(np.int64))
        return {
            'input_ids': sample_tensor[:-1],
            'labels': sample_tensor[1:]
        }
