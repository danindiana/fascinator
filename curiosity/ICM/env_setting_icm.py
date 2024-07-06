# Config #
ENV="MountainCar-v0"
RANDOM_SEED=1

# random seed (reproduciblity)
np.random.seed(RANDOM_SEED)

# set the env
env=gym.make(ENV) # env to import
env.seed(RANDOM_SEED)
env.reset() # reset to env

goal_steps=201 # N episodes per game
training_games=100 # N games to train
