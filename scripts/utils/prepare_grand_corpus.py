import os
import sys

def main():
    print("=" * 60)
    print("     M0.2-Hybrid: Grand Spanish Corpus Compiler")
    print("=" * 60)

    raw_dir = "data/raw_text"
    info_dir = "data/Info_Models"
    output_dir = "data/splits_final"
    os.makedirs(output_dir, exist_ok=True)

    combined_text = ""

    # 1. Add classic and colloquial corpora
    corpora_files = ["quijote.txt", "becquer.txt", "chilean_corpus.txt", "tinyshakespeare.txt"]
    for f_name in corpora_files:
        path = os.path.join(raw_dir, f_name)
        if os.path.exists(path):
            print(f"Adding corpus: {path}")
            with open(path, "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n"

    # 2. Add encyclopedic and technical info documents
    if os.path.exists(info_dir):
        for f_name in os.listdir(info_dir):
            if f_name.endswith(".txt"):
                path = os.path.join(info_dir, f_name)
                print(f"Adding knowledge document: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    combined_text += f.read() + "\n"

    total_chars = len(combined_text)
    print(f"\nCompilation finished! Total size: {total_chars:,} characters.")

    # 3. Save combined text as dataset source
    target_text_path = os.path.join(output_dir, "tinyshakespeare.txt")
    with open(target_text_path, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"Combined corpus saved to: {target_text_path}")

    # 4. Copy stable final 8k tokenizer JSON to the split folder
    src_tok = "data/tokenizer_final_8k.json"
    if not os.path.exists(src_tok):
        src_tok = "data/tokenizers/tokenizer_final_8k.json"
        
    dest_tok = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(src_tok):
        import shutil
        shutil.copy(src_tok, dest_tok)
        print(f"Tokenizer copied successfully to: {dest_tok}")
    else:
        print("Warning: Base 8K tokenizer file not found! Please check your data/ folders.")

if __name__ == "__main__":
    main()
