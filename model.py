from layers import *


class TorcsNet:
    def __init__(self, name, ob_space, ac_space):
        print('obs space: {}'.format(ob_space))
        print('action space: {}'.format(ac_space))

        self.img_height = ob_space[0]
        self.img_width = ob_space[1]
        self.img_channel = ob_space[2]

        self.action_dim = ac_space[0]

        with tf.variable_scope(name+"/Pnet") as Pnet:
            self.__create_input_placeholder()
            self.__create_model()

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, Pnet.name)

    def __create_input_placeholder(self):
        """ all inputs here """
        self.img = tf.placeholder(dtype=tf.float32, shape=[None, self.img_width, self.img_height, self.img_channel])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])

    def __create_model(self):
        """ global conv-layer """
        self.conv1 = conv2d(name="conv1", x=self.img, num_filters=32, filter_size=(3, 3), stride=(1, 1))
        self.conv2 = conv2d(name="conv2", x=self.conv1, num_filters=32, filter_size=(3, 3), stride=(1, 1))
        self.pool1 = maxpool(x=self.conv2, poolsize=(2, 2), stride=(2, 2))
        self.conv3 = conv2d(name="conv3", x=self.pool1, num_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.conv4 = conv2d(name="conv4", x=self.conv3, num_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.pool2 = maxpool(x=self.conv4, poolsize=(2, 2), stride=(2, 2))
        self.flatten_conv = flatten(self.pool2)
        self.fc1 = linear(x=self.flatten_conv, size=512, name="fc1")
        # self.fc1 = tf.nn.dropout(x=self.fc1, keep_prob=0.75)
        self.fc2 = linear(x=self.fc1, size=256, name="fc2")

        self.res = linear(x=self.fc2, size=self.action_dim, name="res")
        self.val = linear(x=self.fc2, size=1, name='val')

    def act(self, obs, sess):
        feed_dict = {self.img: [obs[0]]}
        action = sess.run(self.res, feed_dict=feed_dict)
        return action


if __name__ == "__main__":
    net = TorcsNet('', ob_space=(64, 64, 3), ac_space=(1,))
