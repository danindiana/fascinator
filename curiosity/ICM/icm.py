## BUILD ICM ##
    def inverse_model(self, output_dim=3):
        """
        Predict the action (a_t)
        via the current and next states (s_t, s_t+1)
        """
        def func(f_t, f_t1): 
            #f_t, f_t1 describe the feature vector of s_t and s_t+1, respectively
            inverse_net=concatenate([f_t, f_t1])
            inverse_net=Dense(24, activation='relu')(inverse_net)
            inverse_net=Dense(output_dim, activation='sigmoid')(inverse_net)
            return inverse_net
        return func

    def forward_model(self, output_dim=2):
        """
        Predict the next state (s_t+1)
        via the current state and sction  (s_t, a_t)
        """
        def func(f_t, a_t):
            #f_t describe the feature vector of s_t
            forward_net=concatenate([f_t, a_t])
            forward_net=Dense(24, activation='relu')(forward_net)
            forward_net=Dense(output_dim, activation='linear')(forward_net)
            return forward_net
        return func

     def create_feature_vector(self, input_shape):

        model=Sequential()
        model.add(Dense(24, input_shape=input_shape, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(2, activation='linear', name='feature'))

    def build_icm_model(self, state_shape=(2,), action_shape=(3,)):
        ## Main ICM network
        s_t=Input(shape=state_shape, name="state_t") # (2,)
        s_t1=Input(shape=state_shape, name="state_t1") # (2,)
        a_t=Input(shape=action_shape, name="action") # (3,)

        reshape=Reshape(target_shape= (2,))

        feature_vector_map=self.create_feature_vector((2,))
        fv_t=feature_vector_map(reshape(s_t))
        fv_t1=feature_vector_map(reshape(s_t1))

        a_t_hat=self.inverse_model()(fv_t, fv_t1)
        s_t1_hat=self.forward_model()(fv_t, a_t)

        # the intrinsic reward refelcts the diffrence between
        # the next state versus the predicted next state
        # $r^i_t = \frac{\nu}{2}\abs{\hat{s}_{t+1}-s_{t+1})}^2$
        int_reward=Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                     output_shape=(1,),
                     name="reward_intrinsic")([fv_t1, s_t1_hat])

        #inverse model loss
        inv_loss=Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), 
                                         axis=-1),
                    output_shape=(1,))([a_t, a_t_hat])

        # combined model loss - beta weighs the inverse loss against the
        # rwd (generate from the forward model)
        loss=Lambda(lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1],
                    output_shape=(1,))([int_reward, inv_loss])
        #
        # lmd is lambda, the param the weights the importance of the policy
        # gradient loss against the intrinsic reward
        rwd=Input(shape=(1,))
        loss=Lambda(lambda x: (-self.lmd * x[0] + x[1]), 
                    output_shape=(1,))([rwd, loss])

        return Model([s_t, s_t1, a_t, rwd], loss)
