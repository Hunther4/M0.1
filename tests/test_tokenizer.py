import os
import tempfile
import pytest
from src.tokenizer.bpe import Tokenizer

def test_tokenizer_initialization():
    tokenizer = Tokenizer()
    assert len(tokenizer.vocab) == 258
    assert tokenizer.special_tokens == {"<|endoftext|>": 256, "<|pad|>": 257}
    assert tokenizer.vocab[256] == b"<|endoftext|>"
    assert tokenizer.vocab[257] == b"<|pad|>"
    assert tokenizer.vocab[0] == b"\x00"
    assert tokenizer.vocab[255] == b"\xff"

def test_tokenizer_training_on_dummy_data():
    tokenizer = Tokenizer()
    dummy_text = "abababac"
    tokenizer.train(dummy_text, vocab_size=260, show_progress=False)
    
    assert len(tokenizer.merges) == 2
    assert (97, 98) in tokenizer.merges
    first_merge_id = tokenizer.merges[(97, 98)]
    assert first_merge_id == 258
    assert tokenizer.vocab[258] == b"ab"

def test_tokenizer_lossless_roundtrip():
    tokenizer = Tokenizer()
    training_text = "This is a simple training text with some emojis like 🚀 and 🦄."
    tokenizer.train(training_text, vocab_size=280, show_progress=False)

    test_strings = [
        "Hello World!",
        "Testing UTF-8 roundtrip 🚀🦄.",
        "A string with weird punctuation: !@#$%^&*()_+=-`~[]\\{}|;':\",./<>?",
        "Newline\nand\ttabs\r.",
        ""
    ]

    for s in test_strings:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        assert decoded == s

def test_tokenizer_special_tokens():
    tokenizer = Tokenizer()
    tokenizer.train("simple training text", vocab_size=260, show_progress=False)

    text_with_specials = "hello <|endoftext|> world <|pad|>"
    encoded = tokenizer.encode(text_with_specials, allowed_special="all")
    
    assert 256 in encoded
    assert 257 in encoded
    
    decoded = tokenizer.decode(encoded)
    assert decoded == text_with_specials

    encoded_raw = tokenizer.encode(text_with_specials, allowed_special="none")
    assert 256 not in encoded_raw
    assert 257 not in encoded_raw
    
    decoded_raw = tokenizer.decode(encoded_raw)
    assert decoded_raw == text_with_specials

def test_tokenizer_save_load():
    tokenizer = Tokenizer()
    dummy_text = "hello world hello world hello world"
    tokenizer.train(dummy_text, vocab_size=265, show_progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(filepath)
        assert os.path.exists(filepath)

        new_tokenizer = Tokenizer()
        new_tokenizer.load(filepath)

        assert new_tokenizer.vocab == tokenizer.vocab
        assert new_tokenizer.merges == tokenizer.merges
        assert new_tokenizer.special_tokens == tokenizer.special_tokens

        test_str = "hello world! 🦄"
        assert new_tokenizer.encode(test_str) == tokenizer.encode(test_str)
