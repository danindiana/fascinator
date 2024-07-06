"machine learning" as a concept in computer science and paradigm is revelatory. 

What other kinds of paradigms might be in a similar league? 

machine curiosity? machine novelty-seeking?

How might we implement machine curiosity given the current software stack?

The concept of "machine learning" has indeed been revolutionary in computer science, enabling computers to learn from data and improve their performance over time. Other paradigms in a similar league that have significantly influenced the field include:

Deep Learning: An extension of machine learning that uses deep neural networks to model and solve complex problems.

Reinforcement Learning: A type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties.

Evolutionary Algorithms: These are optimization techniques based on the principles of natural selection and genetics.

Neuroevolution: A combination of neural networks and evolutionary algorithms, where neural networks are evolved rather than explicitly programmed.

Swarm Intelligence: Inspired by the behavior of social insects, this paradigm involves the coordinated behavior of decentralized, self-organized systems.

Regarding "machine curiosity" or "machine novelty-seeking," these concepts are part of the broader field of intrinsic motivation in artificial intelligence. Intrinsic motivation is a research area that explores how agents can be motivated to explore their environment and learn new skills without explicit external rewards.

Implementing Machine Curiosity
Implementing machine curiosity can be approached in several ways, depending on the specific definition and context. Here are some methods that have been explored:

Exploration-Exploitation Trade-off: In reinforcement learning, agents often face a trade-off between exploiting known strategies to maximize rewards and exploring new strategies that might yield higher future rewards. Techniques like ε-greedy policies, softmax exploration, and upper confidence bound (UCB) strategies can be used to balance this trade-off.

Curiosity-Driven Exploration: Curiosity can be modeled as a reward signal for exploring novel states. One popular approach is the Intrinsic Curiosity Module (ICM) proposed by Pathak et al. (2017). ICM uses an inverse dynamics model to predict the action taken between two states and a forward model to predict the next state. The error in the forward model is used as an intrinsic reward signal to encourage exploration of novel states.

Random Network Distillation (RND): This method, introduced by Burda et al. (2018), uses a neural network to predict the output of a fixed random neural network. The prediction error is used as an intrinsic reward signal to encourage exploration of novel states.

Variational Information Maximization for Intrinsic Motivation: This approach, proposed by Mohamed and Rezende (2015), maximizes the mutual information between the agent's actions and the environment's responses, encouraging the agent to explore states that provide the most information.

Novelty Search: In evolutionary algorithms, novelty search encourages the evolution of behaviors that are novel with respect to previously encountered behaviors. This can be achieved by maintaining a history of behaviors and rewarding individuals that exhibit novel behaviors.
