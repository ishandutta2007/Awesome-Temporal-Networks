# Temporal Attention Sinks

Maintaining attention context indefinitely without exploding cache size.

## Overview
Discovers that the initial 2-4 tokens act as attention anchors, absorbing high attention values.

## Architectural Diagram
```mermaid
graph TD
    Tokens[Tokens: 0, 1, 2 ... T] --> Cache[KV Cache]
    Cache --> Sink[Attention Sinks: Fixed Initial Tokens]
    Cache --> Sliding[Sliding Window: Active local context]
```

## Key Mechanisms
- **Streaming Cache:** Keeps initial tokens and a sliding window.
- **No Performance Drop:** Avoids performance collapse on extremely long sequences.

[Back to README](../README.md)
