# The Sequential Recurrence Era (RNN / LSTM)

The sequential recurrence era marks the foundation of temporal modeling in deep learning.

## Overview
Sequential recurrence models process input sequences step-by-step, maintaining a hidden state that acts as the network's memory.

## Architectural Diagram
```mermaid
graph LR
    X_t[Input X_t] --> Cell[Recurrent Cell / LSTM]
    H_prev[Hidden State H_t-1] --> Cell
    Cell --> H_t[Hidden State H_t]
    Cell --> Y_t[Output Y_t]
```

## Key Mechanisms
- **Step-by-step updates:** The hidden state is updated iteratively.
- **Gating mechanisms (LSTM):** Introduced input, forget, and output gates to solve the vanishing gradient problem.

[Back to README](../README.md)
