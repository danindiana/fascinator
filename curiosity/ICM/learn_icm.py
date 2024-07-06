 def learn(self, prev_states, states, actions, rewards):
    #batch train the network
    s_t=prev_states
    s_t1=states
  
    actions=np.array(actions)
  
    icm_loss=self.model.train_on_batch([s_t, s_t1,
                                    np.array(actions),
                                        np.array(rewards).reshape((-1, 1))],
                                        np.zeros((self.batch_size,)))
