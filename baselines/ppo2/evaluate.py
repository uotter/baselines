import os,time
from tqdm import tqdm
import functools, joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.gail.statistics import stats
from baselines.common.tf_util import initialize, FileWriter
import pickle as pkl
from baselines.ppo2_mujoco.model import Model


def evaluate(args,env,network,total_evaluate_trajs,
             eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,Discriminator=None,
             vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
             log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
             save_interval=100, load_path=None, logdir=None,sample_expert=None,expert_name = None,
             checkpointdir=None, **network_kwargs):
    # if Discriminator:
    #     disciminator = Discriminator
    seed = 0
    set_global_seeds(seed)
    total_timesteps = int(total_evaluate_trajs)
    policy = build_policy(env, network, **network_kwargs)
    print('[evaluate] build policy success')
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    model = make_model()

    if checkpointdir is not None:
        model.load(checkpointdir)
        print('[evaluate] load policy success')

    # obs, obs_unchanged = env.reset()
    obs = env.reset()
    states = model.initial_state
    nenv = env.num_envs
    dones = [False for _ in range(nenv)]
    # print('[evaluate]----obs',obs,'-----states',states,'---dones',dones)

    traj_count = 0

    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env or 1, 2 * nlstm)), np.zeros((1))

    extra_args= {}
    state, dones = initialize_placeholders(**extra_args)
    expert_data={}
    reward = []
    ep_obs = []
    ep_obs_unchanged = []
    ep_acs=[]
    epinfos = []
    ep_action_info = []
    ep_action_infos = []
    reward_buffer=deque(maxlen=50)
    lenth_buffer = deque(maxlen=50)
    # while True:
    # for traj_count in range(total_evaluate_trajs):
    while traj_count<=total_evaluate_trajs:
        ep_obs.append(list(obs.reshape(-1)))
        # ep_obs_unchanged.append(list(obs_unchanged.reshape(-1)))
        actions, _, states, _ = model.greedy_step(obs,S=states,M=dones)
        ep_acs.append(list(actions.reshape(-1)))
        # print("current power", obs[0][-1],"ac",actions)
        obs, rew, dones, infos = env.step(actions)
        # obs_unchanged = infos[0]['obs_unchanged']
        for info in infos:
            # maybe_actioninfo = info['action_usable']
            ep_action_info.append(info['action_usable'])
            # if not info['action_usable']:
            #     print("-----action is not usable")
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
                print(maybeepinfo['r'])

        reward.append(rew)
        env.render()
        dones = dones.any() if isinstance(dones, np.ndarray) else dones
        if dones:
            ep_lenth = len(ep_obs)
            print(ep_lenth)
            print(len(ep_action_info),sum(ep_action_info))
            ep_action_infos.append(sum(ep_action_info)/len(ep_action_info))
            # print(env.get_total_action_usable_count())
            obs = env.reset()
            # obs,obs_unchanged = env.reset()
            reward_buffer.extend(epinfos)
            lenth_buffer.extend([ep_lenth])
            reward = []
            traj_count += 1
            # assert len(ep_obs_unchanged)== len(ep_obs)
            if "obs" not in expert_data.keys():
                expert_data["obs"] = ep_obs
            else:
                expert_data["obs"].extend(ep_obs)
            if "acs" not in expert_data.keys():
                expert_data["acs"] = ep_acs
            else:
                expert_data["acs"].extend(ep_acs)
            # if "obs_unchanged" not in expert_data.keys():
            #     expert_data["obs_unchanged"] = ep_obs_unchanged
            # else:
            #     expert_data["obs_unchanged"].extend(ep_obs_unchanged)
            ep_obs=[]
            ep_acs=[]
            ep_action_info = []
            ep_obs_unchanged=[]
            # ep_action_info = []
    print("return length", len(epinfos), safemean([epinfo['r'] for epinfo in reward_buffer]), np.std([epinfo['r'] for epinfo in reward_buffer]))
    # print(safemean([epinfo['r'] for epinfo in reward_buffer]))
    print("average epsiode return", safemean([epinfo['r'] for epinfo in reward_buffer]))
    print("average epsiode OpenAI return", safemean([epinfo['r_env'] for epinfo in reward_buffer]))
    print("average epsiode lenth", safemean([lenth for lenth in lenth_buffer]))

    print("total action success ratio",np.mean(ep_action_infos),np.std(ep_action_infos))
    # print(np.asarray(expert_data["obs"]).shape)
    if sample_expert == 'sample_expert':
        os.makedirs('expert',exist_ok=True)
        pkl.dump(expert_data, open(expert_name + ".pkl", "wb"))
        print('write expert success')
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)