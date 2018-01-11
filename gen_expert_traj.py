import tensorlayer as tl
from gym_torcs import TorcsEnv
import numpy as np
from collections import namedtuple
import h5py


class ETraj:
    def __init__(self, max_episode_len=1000):
        self.img_dim = [64, 64, 3]
        self.n_action = 1  # steer only (float, left and right 1 ~ -1)
        self.max_episode_len = max_episode_len  # maximum step for a game; if crash, get reward -1 and the game restart

        # set up env
        self.env = TorcsEnv(vision=True, throttle=False)

    @staticmethod
    def expert_optimal_action(obs):
        """ Compute steer from image for getting data of demonstration """
        steer = obs.angle * 10 / np.pi
        steer -= obs.trackPos * 0.10
        return np.array([steer])

    @staticmethod
    def expert_noisy_action(obs, noise):
        """ Compute steer from image for getting data of demonstration """
        steer = obs.angle * 10 / np.pi
        steer -= obs.trackPos * 0.10
        steer = np.random.normal(steer, noise)
        return np.array([steer])

    @staticmethod
    def img_reshape(input_img):
        """ (3, 64, 64) --> (64, 64, 3) """
        _img = np.transpose(input_img, (1, 2, 0))
        _img = np.flipud(_img)
        # _img = np.reshape(_img, (1, self.img_dim[0], self.img_dim[1], self.img_dim[2]))
        return _img

    def gen_one_episode(self, noise=0.2):
        print("#"*50)
        print('Collect trajectory...')
        traj = namedtuple('traj', ['img_list', 'action_list', 'reward_list'])
        img_list = list()
        action_list = list()
        reward_list = list()

        ob = self.env.reset(relaunch=True)
        for i in range(self.max_episode_len):
            if i == 0:
                act = np.array([0.0])
            else:
                act = self.expert_noisy_action(ob, noise)

            ob, reward, done, _ = self.env.step(act)
            img_list.append(self.img_reshape(ob.img))
            action_list.append(act)
            reward_list.append(np.array([reward]))
            print("step: {}, action: {}, reward: {}, ob: {}".format(i, act, reward, np.shape(ob.img)))
            if reward < 0:
                print("Crash! Game terminates!")
                break

        traj.img_list = img_list
        traj.action_list = action_list
        traj.reward_list = reward_list

        return traj

    @staticmethod
    def save_data(traj, path):
        # save data
        print("#"*50)
        print('Packing data into arrays... ')
        tl.files.exists_or_mkdir('data/')
        f = h5py.File(path, "w")
        f.create_dataset('img', data=np.array(traj.img_list, dtype='f'), compression="gzip")
        f.create_dataset('action', data=np.array(traj.action_list, dtype='f'), compression="gzip")
        f.create_dataset('reward', data=np.array(traj.reward_list, dtype='f'), compression="gzip")

        # save some teacher's observaion
        tl.files.exists_or_mkdir('image/teacher', verbose=True)
        for i in range(0, len(traj.img_list), 10):
            tl.vis.save_image(traj.img_list[i], 'image/teacher/im_%d.png' % i)

    def close(self):
        # close game
        self.env.end()


if __name__ == "__main__":
    et = ETraj(max_episode_len=1000)
    for noise in [10, 5, 1, 0]:
        for epoc in range(10):
            rand_id = np.random.rand()
            traj = et.gen_one_episode(noise=noise)
            et.save_data(traj=traj, path='data/'+str(noise)+'_'+str(rand_id)+'.h5')

    et.close()
