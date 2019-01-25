#!/usr/bin/env python3
import os, sys

sys.path.append('/home/netease/code/baseline')
if '/home/netease/code/openai_baselines/baselines' in sys.path:
    sys.path.remove('/home/netease/code/openai_baselines/baselines')
import time
from baselines.ppo2.policies import MlpPolicy, MlpAttentionPolicy, MlpDotAttentionPolicy
from baselines import bench, logger
from baselines.common.cmd_util import mujoco_arg_parser


def train(env_id, num_timesteps, seed, method):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2

    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env

    tf.reset_default_graph()

    with tf.Session(config=config):
        model_name = env_id
        init_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        ################# some parameters ################################
        lr = 3e-4
        attention_ent_coef = 0.0
        sigmoid = False
        clip_reward = False
        weak = False
        dot_attention = False
        deep_attention = False
        fix_str = "loss-fix"
        if clip_reward:
            fix_str += "_clip"
        if weak:
            fix_str += "_weak"
        if dot_attention:
            fix_str += "_dot"
        if deep_attention:
            fix_str += "_deep"
        fix_str += "_jump"
        ################# some parameters ################################
        if method == "Attention":
            save_path = model_name + "_" + method + "_" + init_time + "_Sigmoid-" + str(sigmoid) + "_entcoef-" + str(attention_ent_coef) + "_lr" + str(lr) + "_" + fix_str
        else:
            save_path = model_name + "_" + method + "_lr" + str(lr) + "_" + init_time
        tb_dir = os.path.join("/home/netease/data/save/baseline/logs/%s" % save_path)
        logger_dir = os.path.join("/home/netease/data/save/baseline/logger/%s" % save_path)
        for dir in [tb_dir, logger_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        logger.configure(dir=logger_dir)
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)
        set_global_seeds(seed)
        if method == "Attention":
            if dot_attention:
                policy = MlpDotAttentionPolicy
            else:
                policy = MlpAttentionPolicy
        else:
            policy = MlpPolicy
        tb_writer = tf.summary.FileWriter(tb_dir, tf.get_default_session().graph)
        ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                   lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                   ent_coef=0.0,
                   lr=lr,
                   cliprange=0.2,
                   total_timesteps=num_timesteps, attention_ent_coef=attention_ent_coef,
                   writer=tb_writer, save_path=save_path, sigmoid_attention=sigmoid, clip=clip_reward, weak=weak, deep=deep_attention)


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--attention', help='attention or not', type=str, default="Attention")
    args = parser.parse_args()
    print("Going to train.")
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, method=args.attention)


if __name__ == '__main__':
    for _ in range(5):
        main()
