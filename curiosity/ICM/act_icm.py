  def act(self, current_state):
        '''
        this is a less scalabe solution for the implict policy train model below

        This method samples the action space, makes a move, and 
        using the ICM minimizes
        the loss of the combined ext reward and int_reward
        '''
        losses=[]
        for action_option in range(3):

            copy_env=copy.deepcopy(self.env)
            new_state, reward, _, _ = copy_env.step(action_option)
            action_option=self.one_hot_encode_action(action_option)


            loss=self.model.predict([np.array(current_state).\
                              reshape(-1,len(current_state)),
                                        np.array(new_state).\
                              reshape(-1,len(new_state)),
                                        np.array(action_option).\
                              reshape(-1,len(action_option)),
                                        np.array(reward).reshape(-1,1)])
            losses.append(loss)

        chosen_action=np.argmax(losses)

        return chosen_action
   
   
  def make_train_policy_net_model(self, state_shape=(2,), action_shape=(3,)):
        '''
        use the ICM to train the policy models
        Currently not working due to an issue with building 
        dynamic models in Keras
        >> thus not used.
        '''
        current_state=Input(shape=state_shape, name="state_t") # (2,)

        current_action=Lambda(lambda state_t: self.policy_net.predict(state_t),
            output_shape=(3,),
            name="predict_action")(current_state)

        next_state, reward, _, _ =self.env.step(current_action)

        loss=Lambda(lambda s_t, s_t1, a_t, rwd: self.model.\
                        predict([s_t, s_t1, a_t, rwd]),
                        output_shape=(1,),trainable=False)([current_state, 
                          next_state, current_action, reward])


        return Model(current_state, loss)
