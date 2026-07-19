# inference-profiling Specification

## Purpose

This spec defines capabilities for profiling the inference process to track speed (tokens/sec) and memory consumption (VRAM).

## Requirements

### Requirement: Inference Metrics Collection

The system MUST provide a profiling wrapper or CLI mode that captures inference speed (tokens per second) and peak VRAM usage during generation.

#### Scenario: Metrics Capture

- GIVEN a model ready for generation,
- WHEN profiling is enabled,
- THEN the system MUST log or return tokens per second (TPS) and peak VRAM usage.
