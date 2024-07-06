    def one_hot_encode_action(self, action):
        #from int to one hot encode
        action_encoded=np.zeros(self.action_shape, np.float32)
        action_encoded[action]=1
        return action_encoded
