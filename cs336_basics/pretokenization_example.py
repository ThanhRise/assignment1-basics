import os
from typing import BinaryIO, List


# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
#     """
#     Chunk the file into parts that can be counted independently.
#     May return fewer chunks if the boundaries end up overlapping.
#     """
#     assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

#     # Get total file size in bytes
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks

#     # Initial guesses for chunk boundary locations, uniformly spaced
#     # Chunks start on previous index, don't include last index
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)  # Start at boundary guess
#         while True:
#             mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

#             # If EOF, this boundary should be at the end of the file
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break

#             # Find the special token in the mini chunk
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size

#     # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
#     return sorted(set(chunk_boundaries))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: List[bytes],
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    Ensures boundaries align with the first found special token from the list.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list) and all(isinstance(t, bytes) for t in split_special_tokens), \
        "split_special_tokens must be a list of bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Use max(1, ...) to avoid division by zero if desired_num_chunks is 0
    chunk_size = file_size // max(1, desired_num_chunks)

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # Skip the first boundary (0) and the last boundary (file_size)
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        
        found_boundary = False
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                found_boundary = True
                break

            # Find the EARLIEST occurrence of ANY special token in this mini_chunk
            first_occurrence_index = -1
            
            for token in split_special_tokens:
                pos = mini_chunk.find(token)
                if pos != -1:
                    # If we haven't found a token yet, or this one appears earlier
                    if first_occurrence_index == -1 or pos < first_occurrence_index:
                        first_occurrence_index = pos

            # If we found at least one token, update the boundary
            if first_occurrence_index != -1:
                chunk_boundaries[bi] = initial_position + first_occurrence_index
                found_boundary = True
                break
            
            # If nothing found, move the window forward
            initial_position += mini_chunk_size
            
            # Safety break: if we've searched too far (e.g., past the file end or unreasonable distance), stop
            if file.tell() >= file_size:
                chunk_boundaries[bi] = file_size
                found_boundary = True
                break

    # Make sure all boundaries are unique and sorted
    return sorted(list(set(chunk_boundaries)))
