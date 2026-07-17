# BPE Tokenizer Specification

## Purpose
Define a pure-Python Byte-level BPE tokenizer including vocabulary initialization, merge learning, encoding, and decoding.

## Requirements
* The tokenizer MUST initialize with 256 byte tokens and 2 special tokens (`<|endoftext|>`=256, `<|pad|>`=257).
* Pre-tokenization MUST split text using the GPT-2 regex pattern before byte-level processing.
* The trainer MUST merge the most frequent byte/token pairs up to a defined vocabulary size (max 32,768).
* The tokenizer MUST save and load vocabulary and merge rules in JSON format.
* Encoding and decoding operations MUST support loss-less roundtripping for arbitrary strings.

## Scenarios
* **Scenario: Tokenizer Training**
  * GIVEN raw text input and a target vocabulary size of 512,
  * WHEN the tokenizer is trained,
  * THEN it MUST produce 256 merges and output a valid JSON configuration.
* **Scenario: Roundtrip Integrity**
  * GIVEN a trained BPE model and a UTF-8 string containing special tokens,
  * WHEN the string is encoded to tokens and then decoded back,
  * THEN the decoded string MUST match the input exactly.
