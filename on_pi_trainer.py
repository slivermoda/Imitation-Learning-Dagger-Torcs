from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym, logging
from baselines import logger
from baselines import bench
from baselines.common.atari_wrappers import wrap_deepmind
from gym_torcs import TorcsEnv
from model import TorcsNet


def train(num_timesteps, seed):
    from baselines.trpo_mpi.nosharing_cnn_policy import CnnPolicy
    from baselines.trpo_mpi import trpo_mpi
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = TorcsEnv(vision=True, throttle=False)

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return TorcsNet(name=name, ob_space=(64, 64, 3), ac_space=(1,))
    # env = bench.Monitor(
    #
    #     env, logger.get_dir() and osp.jo
    #
    #                     in(logger.get_dir(), str(rank)))
    # env.seed(workerseed)
    # gym.logger.setLevel(logging.WARN)
    #
    # env = wrap_deepmind(env)
    # env.seed(workerseed)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
        max_timesteps=int(num_timesteps * 1.1), gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
    env.end()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    train(num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
