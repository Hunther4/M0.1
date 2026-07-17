import os
import json
import collections
from typing import List, Dict, Tuple, Set, Union
import regex
from tqdm import tqdm

class Tokenizer:
    def __init__(self) -> None:
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = {"<|endoftext|>": 256, "<|pad|>": 257}
        self.inverse_vocab: Dict[bytes, int] = {}
        
        # Initialize 256 base bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.inverse_vocab[bytes([i])] = i
            
        # Add special tokens
        for name, val in self.special_tokens.items():
            val_bytes = name.encode('utf-8')
            self.vocab[val] = val_bytes
            self.inverse_vocab[val_bytes] = val

    def train(self, text: str, vocab_size: int, show_progress: bool = True) -> None:
        """Trains the tokenizer on raw text to achieve vocab_size (max 32768).
        
        Benchmark: Before O(n*m) per merge (full rebuild), After O(n) amortized (incremental).
        Uses incremental pair frequency tracking with reverse index to only update affected pairs.
        """
        if vocab_size > 32768:
            raise ValueError("vocab_size cannot exceed 32768")
        if vocab_size < 258:
            raise ValueError("vocab_size must be at least 258 to accommodate base vocabulary and special tokens")

        # Re-initialize vocabulary to the base (256 bytes + special tokens)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for name, val in self.special_tokens.items():
            self.vocab[val] = name.encode('utf-8')
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}

        num_merges = vocab_size - 258
        if num_merges <= 0:
            return

        # Pre-tokenize using GPT-2 regex pattern
        split_pattern = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        chunks = split_pattern.findall(text)
        
        # Convert each chunk to initial byte IDs
        # To optimize, keep counts of unique words
        word_freqs = collections.Counter(tuple(chunk.encode('utf-8')) for chunk in chunks if chunk)

        # Build initial pair frequency tracking structures
        # pair_freqs: maps pair -> total frequency (occurrences * word_freq)
        # pair_to_words: maps pair -> set of words containing that pair (reverse index)
        # word_to_pairs: maps word -> Counter of pair -> occurrences in that word
        pair_freqs = collections.Counter()
        pair_to_words = collections.defaultdict(set)
        word_to_pairs = {}
        
        for word, freq in word_freqs.items():
            pair_counts = collections.Counter(zip(word, word[1:]))
            word_to_pairs[word] = pair_counts
            for pair, count in pair_counts.items():
                pair_freqs[pair] += count * freq
                pair_to_words[pair].add(word)

        # Merge loop
        # Benchmark: O(n) amortized per merge vs O(n) per merge before
        # Key insight: only update pairs in words affected by the merge
        iterator = range(num_merges)
        if show_progress:
            iterator = tqdm(iterator, desc="Training BPE")

        for i in iterator:
            # Find most frequent pair from incremental tracking
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] == 0:
                break
                
            # Register new token
            new_token_id = 258 + i
            self.merges[best_pair] = new_token_id
            
            # Reconstruct the byte representation for this new token
            bytes_left = self.vocab[best_pair[0]]
            bytes_right = self.vocab[best_pair[1]]
            new_bytes = bytes_left + bytes_right
            self.vocab[new_token_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_token_id
            
            # Get words affected by this merge (only words containing best_pair)
            affected_words = pair_to_words.get(best_pair, set()).copy()
            
            # Remove old pair contributions from affected words
            for word in affected_words:
                freq = word_freqs[word]
                for pair, count in word_to_pairs.get(word, {}).items():
                    pair_freqs[pair] -= count * freq
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        del pair_freqs[pair]
            
            # Merge the pair in affected words and add new contributions
            for word in affected_words:
                freq = word_freqs[word]
                # Apply merge to create new word
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(new_token_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_tuple = tuple(new_word)
                
                # Update word_freqs
                del word_freqs[word]
                word_freqs[new_word_tuple] = word_freqs.get(new_word_tuple, 0) + freq
                
                # Calculate new pairs for the merged word
                new_pair_counts = collections.Counter(zip(new_word_tuple, new_word_tuple[1:]))
                
                # Update word_to_pairs for the new word
                word_to_pairs[new_word_tuple] = new_pair_counts
                
                # Add new pair contributions
                for pair, count in new_pair_counts.items():
                    pair_freqs[pair] += count * freq
                    pair_to_words[pair].add(new_word_tuple)
                
                # Clean up old word from word_to_pairs
                if word in word_to_pairs:
                    del word_to_pairs[word]

    def encode(self, text: str, allowed_special: Union[str, Set[str]] = "all") -> List[int]:
        """Encodes UTF-8 text into a list of token IDs, handling special tokens."""
        if allowed_special == "all":
            active_specials = set(self.special_tokens.keys())
        elif allowed_special == "none" or not allowed_special:
            active_specials = set()
        elif isinstance(allowed_special, set):
            active_specials = allowed_special & set(self.special_tokens.keys())
        else:
            active_specials = set()

        if not active_specials:
            return self._encode_chunk(text)

        # Create regex pattern for active special tokens
        special_pattern = regex.compile("(" + "|".join(regex.escape(t) for t in active_specials) + ")")
        parts = special_pattern.split(text)
        
        ids = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # This is a special token
                ids.append(self.special_tokens[part])
            else:
                # This is normal text
                if part:
                    ids.extend(self._encode_chunk(part))
        return ids

    def _encode_chunk(self, text: str) -> List[int]:
        split_pattern = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        chunks = split_pattern.findall(text)
        
        ids = []
        for chunk in chunks:
            if not chunk:
                continue
            # Start with individual bytes
            chunk_ids = list(chunk.encode('utf-8'))
            
            while len(chunk_ids) >= 2:
                pairs = list(zip(chunk_ids, chunk_ids[1:]))
                min_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                
                if min_pair not in self.merges:
                    break
                
                merged_id = self.merges[min_pair]
                new_ids = []
                j = 0
                while j < len(chunk_ids):
                    if j < len(chunk_ids) - 1 and (chunk_ids[j], chunk_ids[j+1]) == min_pair:
                        new_ids.append(merged_id)
                        j += 2
                    else:
                        new_ids.append(chunk_ids[j])
                        j += 1
                chunk_ids = new_ids
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back into a UTF-8 string (lossless roundtrip)."""
        decoded_bytes = bytearray()
        inverse_special = {val: name for name, val in self.special_tokens.items()}
        
        for token_id in ids:
            if token_id in self.vocab:
                decoded_bytes.extend(self.vocab[token_id])
            elif token_id in inverse_special:
                decoded_bytes.extend(inverse_special[token_id].encode('utf-8'))
            else:
                raise ValueError(f"Unknown token ID: {token_id}")
        return decoded_bytes.decode('utf-8', errors='replace')

    def save(self, filepath: str) -> None:
        """Saves vocab, merges, and special tokens to a JSON file."""
        vocab_serialized = {}
        for k, v in self.vocab.items():
            is_special = False
            for name, val in self.special_tokens.items():
                if val == k:
                    vocab_serialized[str(k)] = name
                    is_special = True
                    break
            if not is_special:
                vocab_serialized[str(k)] = list(v)

        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        merges_serialized = [[pair[0], pair[1]] for pair, idx in sorted_merges]

        data = {
            "vocab": vocab_serialized,
            "merges": merges_serialized
        }

        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Loads vocab, merges, and special tokens from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {}
        self.merges = {}

        # Reconstruct vocab and special tokens
        for k_str, val in data["vocab"].items():
            k = int(k_str)
            if isinstance(val, list):
                val_bytes = bytes(val)
                self.vocab[k] = val_bytes
                self.inverse_vocab[val_bytes] = k
            elif isinstance(val, str):
                self.special_tokens[val] = k
                val_bytes = val.encode('utf-8')
                self.vocab[k] = val_bytes
                self.inverse_vocab[val_bytes] = k
            else:
                raise ValueError(f"Invalid value in vocab JSON: {val}")

        # Reconstruct merges
        for i, pair_list in enumerate(data["merges"]):
            pair = (pair_list[0], pair_list[1])
            parent_id = 258 + i
            self.merges[pair] = parent_id
