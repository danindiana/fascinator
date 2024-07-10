It appears that the code provided is a simplified version of the Curiosity-Driven Exploration framework using TensorFlow and Keras. The code includes the definition of the policy network, inverse model, forward model, and encoder model. It also includes a training loop where the agent interacts with the environment, collects data, and trains the models.

However, the code is not complete and lacks several important aspects for a full implementation of the Curiosity-Driven Exploration framework. Here are some additional considerations and improvements that could be made:

1. **Experience Replay**: To stabilize and improve learning, you could use an experience replay buffer to store past experiences and sample from them during training.

2. **Target Networks**: For the policy network and the forward model, you could use target networks to stabilize the learning process.

3. **Entropy Regularization**: To encourage exploration, you could add entropy regularization to the policy loss.

4. **Batch Training**: Instead of training the models one sample at a time, you could collect a batch of experiences and train the models on the batch.

5. **Normalization**: Normalize the states and rewards to improve the stability of the learning process.

6. **Logging and Evaluation**: Add logging to track the progress of the training and evaluation to assess the performance of the agent.

Here is an extended version of the code with some of these improvements:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import gym
from collections import deque
import random

# Hyperparameters
learning_rate = 0.001
beta = 0.2
eta = 0.1
lambda_ = 0.1
batch_size = 32
buffer_size = 10000

# Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Policy Network
input_state = Input(shape=(state_dim,))
x = Dense(24, activation='relu')(input_state)
x = Dense(24, activation='relu')(x)
policy_output = Dense(action_dim, activation='softmax')(x)
policy_model = Model(inputs=input_state, outputs=policy_output)
policy_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

# Inverse Model
input_state_t = Input(shape=(state_dim,))
input_state_tp1 = Input(shape=(state_dim,))
concat_states = Concatenate()([input_state_t, input_state_tp1])
x = Dense(24, activation='relu')(concat_states)
x = Dense(24, activation='relu')(x)
inverse_output = Dense(action_dim, activation='softmax')(x)
inverse_model = Model(inputs=[input_state_t, input_state_tp1], outputs=inverse_output)
inverse_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

# Forward Model
input_state_t_encoded = Input(shape=(24,))
input_action = Input(shape=(action_dim,))
concat_input = Concatenate()([input_state_t_encoded, input_action])
x = Dense(24, activation='relu')(concat_input)
x = Dense(24, activation='relu')(x)
forward_output = Dense(24, activation='linear')(x)
forward_model = Model(inputs=[input_state_t_encoded, input_action], outputs=forward_output)
forward_model.compile(loss='mse', optimizer=Adam(learning_rate))

# Encoder Model
encoder_input = Input(shape=(state_dim,))
x = Dense(24, activation='relu')(encoder_input)
encoder_output = Dense(24, activation='linear')(x)
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)

# Experience Replay Buffer
replay_buffer = deque(maxlen=buffer_size)

# Training Loop
for episode in range(1000):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        state_input = state.reshape(1, -1)
        action_probs = policy_model.predict(state_input, verbose=0)
        action = np.random.choice(range(action_dim), p=action_probs[0])
        next_state, extrinsic_reward, done, _ = env.step(action)
        
        state_encoded = encoder_model.predict(state_input, verbose=0)
        next_state_encoded = encoder_model.predict(next_state.reshape(1, -1), verbose=0)
        
        inverse_target = np.zeros((1, action_dim))
        inverse_target[0, action] = 1
        
        replay_buffer.append((state, next_state, action, extrinsic_reward, done, state_encoded, next_state_encoded, inverse_target))
        
        state = next_state
        episode_reward += extrinsic_reward
    
    # Training on batch
    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states, next_states, actions, extrinsic_rewards, dones, state_encodings, next_state_encodings, inverse_targets = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        extrinsic_rewards = np.array(extrinsic_rewards)
        dones = np.array(dones)
        state_encodings = np.array(state_encodings)
        next_state_encodings = np.array(next_state_encodings)
        inverse_targets = np.array(inverse_targets)
        
        # Inverse Model Training
        inverse_model.train_on_batch([states, next_states], inverse_targets)
        
        # Forward Model Training
        predicted_next_state_encodings = forward_model.predict([state_encodings, inverse_targets], verbose=0)
        forward_model.train_on_batch([state_encodings, inverse_targets], next_state_encodings)
        
        # Intrinsic Reward Calculation
        intrinsic_rewards = eta / 2 * np.linalg.norm(predicted_next_state_encodings - next_state_encodings, axis=1) ** 2
        
        # Total Reward
        total_rewards = extrinsic_rewards + intrinsic_rewards
        
        # Policy Training
        policy_targets = np.zeros((batch_size, action_dim))
        for i in range(batch_size):
            policy_targets[i, actions[i]] = total_rewards[i]
        
        policy_model.train_on_batch(states, policy_targets)
    
    print(f"Episode: {episode}, Reward: {episode_reward}")

env.close()
```

This extended version includes an experience replay buffer and batch training, which should help improve the stability and performance of the agent.


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import gym

# Hyperparameters
learning_rate = 0.001
beta = 0.2
eta = 0.1
lambda_ = 0.1

# Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Policy Network
input_state = Input(shape=(state_dim,))
x = Dense(24, activation='relu')(input_state)
x = Dense(24, activation='relu')(x)
policy_output = Dense(action_dim, activation='softmax')(x)
policy_model = Model(inputs=input_state, outputs=policy_output)
policy_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

# Inverse Model
input_state_t = Input(shape=(state_dim,))
input_state_tp1 = Input(shape=(state_dim,))
concat_states = Concatenate()([input_state_t, input_state_tp1])
x = Dense(24, activation='relu')(concat_states)
x = Dense(24, activation='relu')(x)
inverse_output = Dense(action_dim, activation='softmax')(x)
inverse_model = Model(inputs=[input_state_t, input_state_tp1], outputs=inverse_output)
inverse_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

# Forward Model
input_state_t_encoded = Input(shape=(24,))
input_action = Input(shape=(action_dim,))
concat_input = Concatenate()([input_state_t_encoded, input_action])
x = Dense(24, activation='relu')(concat_input)
x = Dense(24, activation='relu')(x)
forward_output = Dense(24, activation='linear')(x)
forward_model = Model(inputs=[input_state_t_encoded, input_action], outputs=forward_output)
forward_model.compile(loss='mse', optimizer=Adam(learning_rate))

# Encoder Model
encoder_input = Input(shape=(state_dim,))
x = Dense(24, activation='relu')(encoder_input)
encoder_output = Dense(24, activation='linear')(x)
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)

# Training Loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        state_input = state.reshape(1, -1)
        action_probs = policy_model.predict(state_input, verbose=0)
        action = np.random.choice(range(action_dim), p=action_probs[0])
        next_state, extrinsic_reward, done, _ = env.step(action)
        
        state_encoded = encoder_model.predict(state_input, verbose=0)
        next_state_encoded = encoder_model.predict(next_state.reshape(1, -1), verbose=0)
        
        # Inverse Model Training
        inverse_target = np.zeros((1, action_dim))
        inverse_target[0, action] = 1
        inverse_model.train_on_batch([state_input, next_state.reshape(1, -1)], inverse_target)
        
        # Forward Model Training
        forward_target = next_state_encoded
        forward_model.train_on_batch([state_encoded, inverse_target], forward_target)
        
        # Intrinsic Reward Calculation
        predicted_next_state_encoded = forward_model.predict([state_encoded, inverse_target], verbose=0)
        intrinsic_reward = eta / 2 * np.linalg.norm(predicted_next_state_encoded - next_state_encoded) ** 2
        
        # Total Reward
        total_reward = extrinsic_reward + intrinsic_reward
        
        # Policy Training
        target = np.zeros((1, action_dim))
        target[0, action] = total_reward
        policy_model.train_on_batch(state_input, target)
        
        state = next_state

env.close()
```








Curiosity-Driven Exploration
Our agent is composed of two subsystems: a reward generator that outputs a curiosity-driven intrinsic reward signal
and a policy that outputs a sequence of actions to maximize that reward signal. In addition to intrinsic rewards,
the agent optionally may also receive some extrinsic reward
from the environment. Let the intrinsic curiosity reward
generated by the agent at time t be r
i
t
and the extrinsic reward be r
e
t
. The policy sub-system is trained to maximize
the sum of these two rewards rt = r
i
t + r
e
t
, with r
e
t mostly
(if not always) zero.
Curiosity-driven Exploration by Self-supervised Prediction
Forward
Model
Inverse
Model
features
features
E
ICM	
st st+1
ri
t ri
t
at st st+1
at at+1
(st) (st+1)
ˆ(st+1) aˆt
ICM	
re
t+1 + ri r t+1
e
t + ri
t
Figure 2. The agent in state st interacts with the environment by executing an action at sampled from its current policy π and ends up in
the state st+1. The policy π is trained to optimize the sum of the extrinsic reward (r
e
t
) provided by the environment E and the curiosity
based intrinsic reward signal (r
i
t
) generated by our proposed Intrinsic Curiosity Module (ICM). ICM encodes the states st, st+1 into the
features φ(st), φ(st+1) that are trained to predict at (i.e. inverse dynamics model). The forward model takes as inputs φ(st) and at
and predicts the feature representation φˆ(st+1) of st+1. The prediction error in the feature space is used as the curiosity based intrinsic
reward signal. As there is no incentive for φ(st) to encode any environmental features that can not influence or are not influenced by the
agent’s actions, the learned exploration strategy of our agent is robust to uncontrollable aspects of the environment.
We represent the policy π(st; θP ) by a deep neural network
with parameters θP . Given the agent in state st, it executes
the action at ∼ π(st; θP ) sampled from the policy. θP is
optimized to maximize the expected sum of rewards,
max
θP
Eπ(st;θP )
[Σtrt] (1)
Unless specified otherwise, we use the notation π(s) to denote the parameterized policy π(s; θP ). Our curiosity reward model can potentially be used with a range of policy
learning methods; in the experiments discussed here, we
use the asynchronous advantage actor critic policy gradient
(A3C) (Mnih et al., 2016) for policy learning. Our main
contribution is in designing an intrinsic reward signal based
on prediction error of the agent’s knowledge about its environment that scales to high-dimensional continuous state
spaces like images, bypasses the hard problem of predicting pixels and is unaffected by the unpredictable aspects of
the environment that do not affect the agent.
2.1. Prediction error as curiosity reward
Making predictions in the raw sensory space (e.g. when
st corresponds to images) is undesirable not only because
it is hard to predict pixels directly, but also because it is
unclear if predicting pixels is even the right objective to
optimize. To see why, consider using prediction error in
the pixel space as the curiosity reward. Imagine a scenario
where the agent is observing the movement of tree leaves
in a breeze. Since it is inherently hard to model breeze,
it is even harder to predict the pixel location of each leaf.
This implies that the pixel prediction error will remain high
and the agent will always remain curious about the leaves.
But the motion of the leaves is inconsequential to the agent
and therefore its continued curiosity about them is undesirable. The underlying problem is that the agent is unaware
that some parts of the state space simply cannot be modeled and thus the agent can fall into an artificial curiosity
trap and stall its exploration. Novelty-seeking exploration
schemes that record the counts of visited states in a tabular
form (or their extensions to continuous state spaces) also
suffer from this issue. Measuring learning progress instead
of prediction error has been proposed in the past as one solution (Schmidhuber, 1991). Unfortunately, there are currently no known computationally feasible mechanisms for
measuring learning progress.
If not the raw observation space, then what is the right feature space for making predictions so that the prediction
error provides a good measure of curiosity? To answer
this question, let us divide all sources that can modify the
agent’s observations into three cases: (1) things that can
be controlled by the agent; (2) things that the agent cannot
control but that can affect the agent (e.g. a vehicle driven
by another agent), and (3) things out of the agent’s control
and not affecting the agent (e.g. moving leaves). A good
feature space for curiosity should model (1) and (2) and be
unaffected by (3). This latter is because, if there is a source
of variation that is inconsequential for the agent, then the
agent has no incentive to know about it.

Self-supervised prediction for exploration
Instead of hand-designing a feature representation for every
environment, our aim is to come up with a general mechanism for learning feature representations such that the prediction error in the learned feature space provides a good
intrinsic reward signal. We propose that such a feature
space can be learned by training a deep neural network with
two sub-modules: the first sub-module encodes the raw
state (st) into a feature vector φ(st) and the second submodule takes as inputs the feature encoding φ(st), φ(st+1)
of two consequent states and predicts the action (at) taken
by the agent to move from state st to st+1. Training this
neural network amounts to learning function g defined as:
aˆt = g

st, st+1; θI

(2)
where, aˆt is the predicted estimate of the action at and the
the neural network parameters θI are trained to optimize,
min
θI
LI (ˆat, at) (3)
where, LI is the loss function that measures the discrepancy between the predicted and actual actions. In case at
is discrete, the output of g is a soft-max distribution across
all possible actions and minimizing LI amounts to maximum likelihood estimation of θI under a multinomial distribution. The learned function g is also known as the inverse dynamics model and the tuple (st, at, st+1) required
to learn g is obtained while the agent interacts with the environment using its current policy π(s).
In addition to inverse dynamics model, we train another
neural network that takes as inputs at and φ(st) and predicts the feature encoding of the state at time step t + 1,
φˆ(st+1) = f

φ(st), at; θF

(4)
where φˆ(st+1) is the predicted estimate of φ(st+1) and the
neural network parameters θF are optimized by minimizing
the loss function LF :
LF

φ(st), φˆ(st+1)

=
1
2
kφˆ(st+1) − φ(st+1)k
2
2
(5)
The learned function f is also known as the forward dynamics model. The intrinsic reward signal r
i
t
is computed
as,
r
i
t =
η
2
kφˆ(st+1) − φ(st+1)k
2
2
(6)
where η > 0 is a scaling factor. In order to generate the
curiosity based intrinsic reward signal, we jointly optimize
the forward and inverse dynamics loss described in equations 3 and 5 respectively. The inverse model learns a feature space that encodes information relevant for predicting
the agent’s actions only and the forward model makes predictions in this feature space. We refer to this proposed
curiosity formulation as Intrinsic Curiosity Module (ICM).
As there is no incentive for this feature space to encode
any environmental features that are not influenced by the
agent’s actions, our agent will receive no rewards for reaching environmental states that are inherently unpredictable
and its exploration strategy will be robust to the presence
of distractor objects, changes in illumination, or other nuisance sources of variation in the environment. See Figure 2
for illustration of the formulation.
The use of inverse models has been investigated to learn
features for recognition tasks (Agrawal et al., 2015; Jayaraman & Grauman, 2015). Agrawal et al. (2016) constructed
a joint inverse-forward model to learn feature representation for the task of pushing objects. However, they only
used the forward model as a regularizer for training the inverse model features, while we make use of the error in
the forward model predictions as the curiosity reward for
training our agent’s policy.
The overall optimization problem that is solved for learning
the agent is a composition of equations 1, 3 and 5 and can
be written as,
min
θP ,θI ,θF
"
− λEπ(st;θP )
[Σtrt] + (1 − β)LI + βLF
#
(7)
where 0 ≤ β ≤ 1 is a scalar that weighs the inverse
model loss against the forward model loss and λ > 0 is
a scalar that weighs the importance of the policy gradient
loss against the importance of learning the intrinsic reward
signal.
