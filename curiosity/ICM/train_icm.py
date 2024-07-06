    def get_intrinsic_reward(self, x):
        ## x -> [prev_state, state, action]
        return K.function([self.model.get_layer("state_t").input,
                       self.model.get_layer("state_t1").input,
                       self.model.get_layer("action").input],
                      [self.model.get_layer("reward_intrinsic").output])(x)[0]
    
    def batch_train(self):
 
        # track results
        states=[]
        actions=[]
        ext_rewards=[]

        game_best_position=-.4
        positions=np.ndarray([0,2])

        for game_index in range(self.training_games):
            # observation >> the current position of Car and its velocity
            state=env.reset() # each game is a new env
            game_reward=0
            running_reward=0
            print(game_index)
            for step_index in range(self.game_steps):
                ## act ##
                # choose a random action at first, otherwise given policy net
                # push left(0), no push(1) and push right(2)
                if step_index>self.batch_size:
                  action=self.act(state)
                else:
                  action=self.env.action_space.sample()
                
                next_state, ext_reward, done, info=self.env.step(action)

                ## track restults ##
                #keep track on step properties (progress)
                action=self.one_hot_encode_action(action)


                ## get reward ##
                
                # prep data to get int reward
                int_r_state=np.reshape(state, (1,2))
                int_r_next_state=np.reshape(next_state, (1,2))
                int_r_action=np.reshape(action, (1,3))
                
                #get intrinsic reward
                int_reward=self.get_intrinsic_reward(
                                        [np.array(int_r_state),
                                         np.array(int_r_next_state),
                                         np.array(int_r_action)])
                
                #calc total reward
                reward=int_reward+ext_reward
                game_reward+=reward

                if state[0]>game_best_position:
                  game_best_position=state[0]
                  positions=np.append(positions, 
                             [[game_index, game_best_position]], axis=0)
                  
                  running_reward+=10
                else:
                  running_reward+=reward

                #move state
                state=next_state
                
                ## batch train ##
                
                # prep data for batch train
                states.append(next_state)
                ext_rewards.append(ext_reward)
                actions.append(action)

                if step_index%self.batch_size==0 and \
                  step_index>=self.batch_size:
                    all_states=states[-(self.batch_size+1):]
                    

                    self.learn(prev_states=all_states[:self.batch_size],
                                states=all_states[-self.batch_size:],
                                actions=actions[-self.batch_size:],
                                rewards=ext_rewards[-self.batch_size:])

                if done:
                    # done if reached 0.5(top) position
                    break

            self.rewards[game_index]=game_reward
            positions[-1][0]=game_index
            self.positions[game_index]=positions[-1]
            print(self.positions[game_index])

        self.show_training_data()
  
  
  
    def show_training_data(self):
        ## plot the training data
        
        self.positions[0]=[0,-0.4]
        plt.figure(1, figsize=[10,5])
        plt.subplot(211)
        plt.plot(self.positions[:,0], self.positions[:,1])
        plt.xlabel('Episode')
        plt.ylabel('Furthest Position')
        plt.subplot(212)
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
