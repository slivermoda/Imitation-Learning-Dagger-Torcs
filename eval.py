import tensorflow as tf
from gym_torcs import TorcsEnv
from model import TorcsNet
import numpy as np


def img_reshape(input_img):
    """ (3, 64, 64) --> (64, 64, 3) """
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, 64, 64, 3))
    return _img


if __name__ == "__main__":
    reward_sum = 0.0
    steps = 1000
    obs_space = (64, 64, 3)
    action_space = (1,)

    model = TorcsNet(obs_space, action_space)
    saver = tf.train.Saver(var_list=model.var_list)
    sess = tf.Session()
    saver.restore(sess, save_path='save/model.ckpt')
    print("model successfully restored.")

    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    ob_list = list()

    for i in range(steps):
        act = model.act(img_reshape(ob.img), sess)
        ob, reward, done, _ = env.step(act)
        if done is True:
            break
        else:
            ob_list.append(ob)
        reward_sum += reward
        print('step: {}, act: {}, reward: {}'.format(i, act, reward))
    print("PLAY WITH THE TRAINED MODEL")
    print(reward_sum)
    env.end()
