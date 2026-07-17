# Token Counter CLI Specification

## Purpose
Provide a command-line tool to count tokens in a text file or string and render them with alternating visual highlights.

## Requirements
* The CLI MUST accept a path to a trained tokenizer JSON file.
* The CLI MUST accept raw text from a file path or direct string arguments.
* The CLI MUST print the total token count to stdout.
* The CLI MUST render individual token boundaries using alternating ANSI escape colors.

## Scenarios
* **Scenario: Display Highlighted Tokens**
  * GIVEN a trained BPE tokenizer model and the input text "Hello world",
  * WHEN the token-counter CLI is executed with these arguments,
  * THEN it MUST output the total token count and print "Hello" and " world" in alternating colors.
