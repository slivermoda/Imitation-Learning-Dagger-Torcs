import tensorflow as tf
import tensorlayer as tl
import numpy as np
from model import TorcsNet
import glob
import h5py as h5


class BCTrainer:  # Behavior cloning
    def __init__(self):
        self.obs_space = (64, 64, 3)
        self.action_space = (1,)
        self.reward_space = (1,)
        self.lr = 0.01
        self.sess = tf.Session()
        self.__create_train_ops()
        self.sess.run(tf.global_variables_initializer())
        self.total_num_batch = 0

        self.saver = tf.train.Saver()

    def __create_train_ops(self):
        self.net = TorcsNet('', ob_space=self.obs_space, ac_space=self.action_space)
        self.loss = tf.losses.mean_squared_error(labels=self.net.action, predictions=self.net.res)

        # gradients
        self.grads = tf.gradients(self.loss, self.net.var_list)
        self.grads, _ = tf.clip_by_global_norm(self.grads, 40.0)
        self.grads_and_vars = list(zip(self.grads, self.net.var_list))

        # the optimizer. each worker has its own
        self.opt = tf.train.RMSPropOptimizer(self.lr, decay=0.99, momentum=0.0, epsilon=0.1, use_locking=False)
        self.train_op = self.opt.apply_gradients(self.grads_and_vars)

        # summary
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", tf.global_norm(self.grads))
        tf.summary.scalar("predicted_action", tf.reduce_mean(self.net.res))
        tf.summary.scalar("real_action", tf.reduce_mean(self.net.action))
        self.summary_op = tf.summary.merge_all()

    def process_one_batch(self, bx, by):
        feed_dict = {self.net.img: bx,
                     self.net.action: by}
        _, err, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict)
        self.total_num_batch += 1
        self.summary_writer.add_summary(summary, self.total_num_batch)
        if self.total_num_batch % 50 == 0:
            self.saver.save(self.sess, "save/model.ckpt")
            print("model saved.")
        return err


class PoRWTrainer:  # Policy re-weighting
    def __init__(self):
        self.obs_space = (64, 64, 3)
        self.action_space = (1,)
        self.reward_space = (1,)
        self.lr = 0.001
        self.sess = tf.Session()
        self.__create_train_ops()
        self.sess.run(tf.global_variables_initializer())
        self.total_num_batch = 0

        self.saver = tf.train.Saver()

    def __create_train_ops(self):
        self.net = TorcsNet('', ob_space=self.obs_space, ac_space=self.action_space)
        self.adv = tf.placeholder(tf.float32, [None, 1], name="adv")
        # self.r = tf.placeholder(tf.float32, [None], name="r")

        self.pi_loss = tf.losses.mean_squared_error(labels=self.net.action, predictions=self.net.res, weights=self.adv)
        # self.pi_loss = self.pi_loss * self.adv
        # value loss: a least square
        # self.vf_loss = 0.5 * tf.reduce_mean(tf.square(self.net.val - self.r))
        self.loss = self.pi_loss  # + self.vf_loss

        # gradients
        self.grads = tf.gradients(self.loss, self.net.var_list)
        self.grads, _ = tf.clip_by_global_norm(self.grads, 40.0)
        self.grads_and_vars = list(zip(self.grads, self.net.var_list))

        # the optimizer. each worker has its own
        self.opt = tf.train.RMSPropOptimizer(self.lr, decay=0.99, momentum=0.0, epsilon=0.1, use_locking=False)
        self.train_op = self.opt.apply_gradients(self.grads_and_vars)

        # summary
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        tf.summary.scalar("pi_loss", self.pi_loss)
        # tf.summary.scalar("vf_loss", self.vf_loss)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", tf.global_norm(self.grads))
        tf.summary.scalar("predicted_action", tf.reduce_mean(self.net.res))
        tf.summary.scalar("real_action", tf.reduce_mean(self.net.action))
        self.summary_op = tf.summary.merge_all()

    def process_one_batch(self, bx, by, br):
        feed_dict = {self.net.img: bx,
                     self.net.action: by,
                     self.adv: br}
        _, err, summary = self.sess.run([self.train_op, self.pi_loss, self.summary_op], feed_dict)
        self.total_num_batch += 1
        self.summary_writer.add_summary(summary, self.total_num_batch)
        if self.total_num_batch % 10 == 0:
            self.saver.save(self.sess, "save_pi_reweight/model.ckpt")
            print("model saved.")
        return err


def accumulate_r(r):
    gamma = 0.999999
    L = np.shape(r)[0]
    R = np.zeros([L, 1])
    discounted_r = 0
    for i in range(L):
        R[L-i-1] = r[L-i-1] + gamma * discounted_r
        discounted_r = R[L-i-1]
    return R


if __name__ == "__main__":
    # data
    data_folder = "data"
    data_list = glob.glob(data_folder+'/*.h5')
    x = None
    y = None
    r = None

    for f in data_list:
        ep_data = h5.File(f, 'r')
        if x is None:
            x = ep_data['img'][:]
            y = ep_data['action'][:]
            r = accumulate_r(ep_data['reward'][:])
        else:
            x = np.concatenate((x, ep_data['img'][:]), axis=0)
            y = np.concatenate((y, ep_data['action'][:]), axis=0)
            r = np.concatenate((r, accumulate_r(ep_data['reward'][:])), axis=0)

    # trainer
    Imitate = True
    Epoches = 5
    if Imitate:
        print('BC training.')
        trainer = BCTrainer()
        batch_size = 10
        for epoc in range(Epoches):
            print("epoch: {}".format(epoc))
            for x_, y_ in tl.iterate.minibatches(x, y, batch_size, shuffle=True):
                x_ /= 255.0
                error = trainer.process_one_batch(x_, y_)
                print("epoch: {}, error: {}".format(epoc, error))
    else:
        trainer = PoRWTrainer()
        batch_size = 10
        for epoc in range(Epoches):
            print("epoch: {}".format(epoc))
            for x_, y_ in tl.iterate.minibatches(x, np.concatenate((y, r), axis=1), batch_size, shuffle=True):
                x_ /= 255.0
                r_ = np.reshape(y_[:, 1], [batch_size, 1])
                y_ = np.reshape(y_[:, 0], [batch_size, 1])
                r_ = np.exp(r_/10000.0*0.05)
                # print(r_)
                error = trainer.process_one_batch(x_, y_, r_)
                print("epoch: {}, error: {}".format(epoc, error))
