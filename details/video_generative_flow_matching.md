# Spatio-Temporal Video Generative Flow-Matching (Sora Class)

Generative modeling of spatio-temporal video sequences.

## Overview
Video sequences are tokenized into 3D spacetime cubes and processed using Diffusion Transformers or Flow-Matching.

## Architectural Diagram
```mermaid
graph TD
    Video[Raw Video Frame Stream] --> Tokens[3D Spacetime Cubes]
    Tokens --> FlowMatch[Flow Matching / DiT Backbone]
    FlowMatch --> GenVideo[Generated Realistic Physics Video]
```

## Key Mechanisms
- **3D Spacetime Tokenization:** Joint spatial and temporal token tracking.
- **Flow Matching:** Straight-path trajectories for faster generation.

[Back to README](../README.md)
