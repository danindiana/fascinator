class ICM::
    def __init__(self, env=env, training_games=training_games, 
                 game_steps=game_steps, beta=0.01, lmd=0.99):
        
        self.env=env #import env
        self.state_shape=env.observation_space.shape # the state space
        self.action_shape=env.action_space.n # the action space
        
        self.lmd=lmd # ratio of the external loss against the intrinsic reward
        self.beta=beta # ratio of the inverse loss against the forward reward 

        self.training_games=training_games #N training games
        self.goal_steps=goal_steps # N training steps
        self.batch_size=10 # batch size for training the model

        self.model=self.build_icm_model() #build ICM
        self.model.compile(optimizer=Adam(), loss="mse") #Complies ICM
        
        self.positions=np.zeros((self.training_games,2)) #record learning process
        self.rewards=np.zeros(self.training_games) #record learning process
