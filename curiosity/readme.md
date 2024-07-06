Explanation
Environment: A simplified environment class that generates random states and rewards.

Agent: An agent class that builds a neural network model, selects actions, and trains the model.

ICM: An intrinsic curiosity module that includes a forward model and an inverse model to compute intrinsic rewards.

Training Loop: The main loop that runs episodes, selects actions, computes rewards, and trains the agent.

This refactored version integrates curiosity-driven exploration into the reinforcement learning framework, following the flowchart provided in the Mermaid diagram.

```mermaid
graph TD
    A[Start] --> B[Initialize Environment]
    B --> C[Initialize Agent]
    C --> D[Initialize ICM]
    D --> E[Episode Loop]
    E --> F[State Loop]
    F --> G[Agent Selects Action]
    G --> H[Environment Steps]
    H --> I[Compute Intrinsic Reward]
    I --> J[Compute Total Reward]
    J --> K[Agent Trains]
    K --> L[Update State]
    L --> F
    E --> M[End Episode]
    M --> E
```
