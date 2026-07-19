import os
import sys
import struct

def read_gguf_string(f):
    length_bytes = f.read(8)
    if not length_bytes:
        return ""
    length = struct.unpack("<Q", length_bytes)[0]
    return f.read(length).decode("utf-8", errors="ignore")

def read_gguf_value(f, val_type):
    # GGUF Value Types:
    # 0 = UINT8, 1 = INT8, 2 = UINT16, 3 = INT16, 4 = UINT32, 5 = INT32,
    # 6 = FLOAT32, 7 = BOOL, 8 = STRING, 9 = ARRAY, 10 = UINT64, 11 = INT64, 12 = FLOAT64
    if val_type == 0:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == 1:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == 2:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == 3:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == 4:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == 5:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == 6:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == 7:
        return struct.unpack("<b", f.read(1))[0] != 0
    elif val_type == 8:
        return read_gguf_string(f)
    elif val_type == 9:
        # Array type
        sub_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(min(array_len, 20)): # limit read
            arr.append(read_gguf_value(f, sub_type))
        # Skip remaining if array is huge
        if array_len > 20:
            # We skip remaining elements based on type size
            type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            if sub_type in type_sizes:
                f.seek(type_sizes[sub_type] * (array_len - 20), 1)
            else:
                # String array fallback: read and discard
                for _ in range(array_len - 20):
                    read_gguf_value(f, sub_type)
        return arr
    elif val_type == 10:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == 11:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == 12:
        return struct.unpack("<d", f.read(8))[0]
    return None

def parse_gguf(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    print(f"Parsing GGUF metadata: {file_path}")
    print("-" * 50)
    
    with open(file_path, "rb") as f:
        # 1. Magic
        magic = f.read(4)
        if magic != b"GGUF":
            print("Error: Invalid GGUF magic header.")
            return
            
        # 2. Version
        version = struct.unpack("<I", f.read(4))[0]
        print(f"GGUF Version: {version}")
        
        # 3. Tensor count & Key-value count
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        
        print(f"Tensors: {tensor_count}")
        print(f"Metadata Key-Value Pairs: {kv_count}")
        print("-" * 50)
        
        # 4. Read KV Pairs
        metadata = {}
        for _ in range(kv_count):
            key = read_gguf_string(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = read_gguf_value(f, val_type)
            metadata[key] = val
            
        # Display key features
        features = [
            "general.architecture",
            "general.name",
            "qwen2.block_count",
            "qwen2.context_length",
            "qwen2.embedding_length",
            "qwen2.feed_forward_length",
            "qwen2.attention.head_count",
            "qwen2.attention.head_count_kv",
            "qwen2.expert_count",
            "qwen2.expert_used_count"
        ]
        
        # Also check for llama or general keys (in case of other mappings)
        for key, val in metadata.items():
            if any(f in key for f in ["architecture", "name", "block_count", "context_length", "expert_count", "head_count"]):
                # limit array printing size
                print(f"{key}: {val[:5]}... (Array of size {len(val)})" if isinstance(val, list) else f"{key}: {val}")
                
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_gguf_metadata.py <gguf_file_path>")
    else:
        parse_gguf(sys.argv[1])
