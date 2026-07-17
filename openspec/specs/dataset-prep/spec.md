# Dataset Preparation Specification

## Purpose
Standardize remote download and local file ingestion of datasets (e.g., TinyShakespeare) for training tokenizers and models.

## Requirements
* The utility MUST support downloading TinyShakespeare from the official source if no local file is provided.
* The utility MUST accept a custom local text file as input.
* The utility MUST output a cleansed single UTF-8 text file to the designated target directory.

## Scenarios
* **Scenario: Remote Download**
  * GIVEN the remote TinyShakespeare source is accessible,
  * WHEN the preparation script is run with no custom path,
  * THEN it MUST download, validate, and save the dataset to disk.
* **Scenario: Custom Ingestion**
  * GIVEN a valid local text file,
  * WHEN the preparation script is run with the file's path,
  * THEN it MUST verify UTF-8 encoding and save the file to the target path.
