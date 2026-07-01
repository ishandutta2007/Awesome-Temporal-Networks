# Rotary Position Embeddings (RoPE)

Geometrical relative position encoding using complex rotations.

## Overview
Instead of adding static vectors, features are rotated based on their time coordinates.

## Architectural Diagram
```mermaid
graph LR
    Input[Query/Key Vector] --> Rot[Rotation Operator e^(i*theta)]
    Pos[Position Offset] --> Rot
    Rot --> Out[Rotated Vector]
```

## Key Mechanisms
- **Relative Distance Representation:** Inner product decays naturally with distance.
- **Linear Translation Invariance:** Retains position relations geometrically.

[Back to README](../README.md)
