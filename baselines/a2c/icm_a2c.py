import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

from collections import deque

skores = []

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, v_mix_coef=0.5, max_grad_norm=0.5, lr_alpha=7e-4, lr_beta=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(50e5), lrschedule='linear',
            r_ex_coef=1.0, r_in_coef=0.0, v_ex_coef=1.0):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch], 'A')
        R_EX = tf.placeholder(tf.float32, [nbatch], 'R_EX')
        ADV_EX = tf.placeholder(tf.float32, [nbatch], 'ADV_EX')
        RET_EX = tf.placeholder(tf.float32, [nbatch], 'RET_EX')
        V_MIX = tf.placeholder(tf.float32, [nbatch], 'V_MIX')
        DIS_V_MIX_LAST = tf.placeholder(tf.float32, [nbatch], 'DIS_V_MIX_LAST')
        COEF_MAT = tf.placeholder(tf.float32, [nbatch, nbatch], 'COEF_MAT')
        LR_ALPHA = tf.placeholder(tf.float32, [], 'LR_ALPHA')
        LR_BETA = tf.placeholder(tf.float32, [], 'LR_BETA')

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        r_mix = r_ex_coef * R_EX + r_in_coef * tf.reduce_sum(train_model.r_in * tf.one_hot(A, nact), axis=1)
        ret_mix = tf.squeeze(tf.matmul(COEF_MAT, tf.reshape(r_mix, [nbatch, 1])), [1]) + DIS_V_MIX_LAST
        adv_mix = ret_mix - V_MIX

        neglogpac = train_model.pd.neglogp(A)
        pg_mix_loss = tf.reduce_mean(adv_mix * neglogpac)
        v_mix_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_mix), ret_mix))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        policy_loss = pg_mix_loss - ent_coef * entropy + v_mix_coef * v_mix_loss

        policy_params = train_model.policy_params
        policy_grads = tf.gradients(policy_loss, policy_params)
        if max_grad_norm is not None:
            policy_grads, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
        policy_grads_and_vars = list(zip(policy_grads, policy_params))
        policy_trainer = tf.train.RMSPropOptimizer(learning_rate=LR_ALPHA, decay=alpha, epsilon=epsilon)
        policy_train = policy_trainer.apply_gradients(policy_grads_and_vars)

        rmss = [policy_trainer.get_slot(var, 'rms') for var in policy_params]
        policy_params_new = {}
        for grad, rms, var in zip(policy_grads, rmss, policy_params):
            ms = rms + (tf.square(grad) - rms) * (1 - alpha)
            policy_params_new[var.name] = var - LR_ALPHA * grad / tf.sqrt(ms + epsilon)
        policy_new = train_model.policy_new_fn(policy_params_new, ob_space, ac_space, nbatch, nsteps)

        neglogpac_new = policy_new.pd.neglogp(A)
        ratio_new = tf.exp(tf.stop_gradient(neglogpac) - neglogpac_new)
        pg_ex_loss = tf.reduce_mean(-ADV_EX * ratio_new)
        v_ex_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_ex), RET_EX))
        intrinsic_loss = pg_ex_loss + v_ex_coef * v_ex_loss

        intrinsic_params = train_model.intrinsic_params
        intrinsic_grads = tf.gradients(intrinsic_loss, intrinsic_params)
        if max_grad_norm is not None:
            intrinsic_grads, intrinsic_grad_norm = tf.clip_by_global_norm(intrinsic_grads, max_grad_norm)
        intrinsic_grads_and_vars = list(zip(intrinsic_grads, intrinsic_params))
        intrinsic_trainer = tf.train.RMSPropOptimizer(learning_rate=LR_BETA, decay=alpha, epsilon=epsilon)
        intrinsic_train = intrinsic_trainer.apply_gradients(intrinsic_grads_and_vars)

        lr_alpha = Scheduler(v=lr_alpha, nvalues=total_timesteps, schedule=lrschedule)
        lr_beta = Scheduler(v=lr_beta, nvalues=total_timesteps, schedule=lrschedule)

        all_params = tf.global_variables()

        def train(obs, policy_states, masks, actions, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat):
            advs_ex = ret_ex - v_ex
            for step in range(len(obs)):
                cur_lr_alpha = lr_alpha.value()
                cur_lr_beta= lr_beta.value()
            td_map = {train_model.X:obs, policy_new.X:obs, A:actions, R_EX:r_ex, ADV_EX:advs_ex, RET_EX:ret_ex,
                      V_MIX:v_mix, DIS_V_MIX_LAST:dis_v_mix_last, COEF_MAT:coef_mat,
                      LR_ALPHA:cur_lr_alpha, LR_BETA:cur_lr_beta}
            if policy_states is not None:
                td_map[train_model.PS] = policy_states
                td_map[train_model.M] = masks
            return sess.run(
                [entropy, policy_train, intrinsic_train],
                td_map
            )[0]

        def save(save_path):
            ps = sess.run(all_params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(all_params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.intrinsic_reward = step_model.intrinsic_reward
        self.init_policy_state = step_model.init_policy_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99, r_ex_coef=1.0, r_in_coef=0.0, no_ex=False, no_in=False):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.policy_states = model.init_policy_state
        self.dones = [False for _ in range(nenv)]
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])
        self.no_ex = no_ex
        self.no_in = no_in

    def run(self):
        mb_obs, mb_r_ex, mb_r_in, mb_ac, mb_v_ex, mb_v_mix, mb_dones = [],[],[],[],[],[],[]
        mb_policy_states = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [],[],[],[]
        for n in range(self.nsteps):
            mb_policy_states.append(self.policy_states)
            ac, v_ex, v_mix, policy_states, _ = self.model.step(self.obs, self.policy_states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_ac.append(ac)
            mb_v_ex.append(v_ex)
            mb_v_mix.append(v_mix)
            mb_dones.append(self.dones)
            obs, r_ex, dones, infos = self.env.step(ac)
            r_in = self.model.intrinsic_reward(self.obs, ac)
            #print(type(r_in))
            if self.no_ex:
                r_ex = np.zeros_like(r_ex)
            if self.no_in:
                r_in = np.zeros_like(r_in)
            mb_r_ex.append(r_ex)
            mb_r_in.append(r_in)
            self.policy_states = policy_states
            self.dones = dones
            self.ep_r_ex += r_ex
            self.ep_r_in += r_in
            self.ep_len += 1
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    ep_info.append(maybeepinfo)
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0
            self.obs = obs
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32).swapaxes(1, 0)
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32).swapaxes(1, 0)
        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        mb_ac = np.asarray(mb_ac, dtype=np.int32).swapaxes(1, 0)
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0)
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_v_ex, last_v_mix = self.model.value(self.obs, self.policy_states, self.dones)
        last_v_ex, last_v_mix = last_v_ex.tolist(), last_v_mix.tolist()
        #discount/bootstrap off value fn
        mb_ret_ex, mb_ret_mix = np.zeros(mb_r_ex.shape), np.zeros(mb_r_mix.shape)
        for n, (r_ex, r_mix, dones, v_ex, v_mix) in enumerate(zip(mb_r_ex, mb_r_mix, mb_dones, last_v_ex, last_v_mix)):
            r_ex, r_mix = r_ex.tolist(), r_mix.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                ret_ex = discount_with_dones(r_ex+[v_ex], dones+[0], self.gamma)[:-1]
                ret_mix = discount_with_dones(r_mix+[v_mix], dones+[0], self.gamma)[:-1]
            else:
                ret_ex = discount_with_dones(r_ex, dones, self.gamma)
                ret_mix = discount_with_dones(r_mix, dones, self.gamma)
            mb_ret_ex[n], mb_ret_mix[n] = ret_ex, ret_mix
        mb_r_ex = mb_r_ex.flatten()
        mb_r_in = mb_r_in.flatten()
        mb_ret_ex = mb_ret_ex.flatten()
        mb_ret_mix = mb_ret_mix.flatten()
        mb_ac = mb_ac.flatten()
        mb_v_ex = mb_v_ex.flatten()
        mb_v_mix = mb_v_mix.flatten()
        mb_masks = mb_masks.flatten()
        mb_dones = mb_dones.flatten()
        return mb_obs, mb_ac, mb_policy_states, mb_r_in, mb_r_ex, mb_ret_ex, mb_ret_mix,\
               mb_v_ex, mb_v_mix, last_v_ex, last_v_mix, mb_masks, mb_dones,\
               ep_info, ep_r_ex, ep_r_in, ep_len

def learn(policy, env, seed, nsteps=5, total_timesteps=int(50e6), v_mix_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr_alpha=7e-4, lr_beta=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100,
          v_ex_coef=1.0, r_ex_coef=0.0, r_in_coef=1.0, no_ex=False, no_in=False):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
                  v_ex_coef=v_ex_coef, max_grad_norm=max_grad_norm, lr_alpha=lr_alpha, lr_beta=lr_beta,
                  alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule,
                  v_mix_coef=v_mix_coef, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, r_ex_coef=r_ex_coef, r_in_coef=r_in_coef, no_ex=no_in, no_in=no_in)

    nbatch = nenvs*nsteps
    tstart = time.time()
    epinfobuf = deque(maxlen=100)
    eprexbuf = deque(maxlen=100)
    eprinbuf = deque(maxlen=100)
    eplenbuf = deque(maxlen=100)
    for update in range(1, total_timesteps//nbatch+1):
        obs, ac, policy_states, r_in, r_ex, ret_ex, ret_mix, \
        v_ex, v_mix, last_v_ex, last_v_mix, masks, dones, \
        epinfo, ep_r_ex, ep_r_in, ep_len = runner.run()
        dis_v_mix_last = np.zeros([nbatch], np.float32)
        coef_mat = np.zeros([nbatch, nbatch], np.float32)
        for i in range(nbatch):
            dis_v_mix_last[i] = gamma ** (nsteps - i % nsteps) * last_v_mix[i // nsteps]
            coef = 1.0
            for j in range(i, nbatch):
                if j > i and j % nsteps == 0:
                    break
                coef_mat[i][j] = coef
                coef *= gamma
                if dones[j]:
                    dis_v_mix_last[i] = 0
                    break
        entropy = model.train(obs, policy_states[0], masks, ac, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat)

        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        epinfobuf.extend(epinfo)
        eprexbuf.extend(ep_r_ex)
        eprinbuf.extend(ep_r_in)
        eplenbuf.extend(ep_len)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("entropy", float(entropy))
            v_ex_ev = explained_variance(v_ex, ret_ex)
            logger.record_tabular("v_ex_ev", float(v_ex_ev))
            v_mix_ev = explained_variance(v_mix, ret_mix)
            logger.record_tabular("v_mix_ev", float(v_mix_ev))
            logger.record_tabular("gamescoremean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("gamelenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
            skores.append(safemean([epinfo['r'] for epinfo in epinfobuf]))
    with open('scores_no_ICM.p' 'wb') as f:
        pickle.dump(skores, f)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)








import os, csv, random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from agents.A2C import Model as A2C_Model
from networks.networks import ActorCriticSMB
from networks.special_units import IC_Features, IC_ForwardModel_Head, IC_InverseModel_Head
from utils.RolloutStorage import RolloutStorage

from timeit import default_timer as timer

class Model(A2C_Model):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym', tb_writer=None):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir, tb_writer=tb_writer)

        self.declare_networks()
            
        if not self.config.recurrent_policy_grad:
            self.optimizer = optim.RMSprop(list(self.model.parameters())+list(self.ICMfeaturizer.parameters())+list(self.ICMForwardModel.parameters())+list(self.ICMBackwardModel.parameters()), lr=self.config.LR, alpha=0.99, eps=1e-5)
        else:
            self.optimizer = optim.Adam(list(self.model.parameters())+list(self.ICMfeaturizer.parameters())+list(self.ICMForwardModel.parameters())+list(self.ICMBackwardModel.parameters()), lr=self.config.LR)
        
        #move to correct device
        self.model = self.model.to(self.config.device)
        self.ICMfeaturizer = self.ICMfeaturizer.to(self.config.device)
        self.ICMForwardModel = self.ICMForwardModel.to(self.config.device)
        self.ICMBackwardModel = self.ICMBackwardModel.to(self.config.device)

        if self.static_policy:
            self.model.eval()
            self.ICMfeaturizer.eval()
            self.ICMForwardModel.eval()
            self.ICMBackwardModel.eval()
        else:
            self.model.train()
            self.ICMfeaturizer.train()
            self.ICMForwardModel.train()
            self.ICMBackwardModel.train()

        self.config.rollouts = RolloutStorage(self.config.rollout, self.config.num_agents,
            self.num_feats, self.env.action_space, self.model.state_size,
            self.config.device, config.USE_GAE, config.gae_tau)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []


    def declare_networks(self):
        self.model = ActorCriticSMB(self.num_feats, self.num_actions, self.config.recurrent_policy_grad, self.config.gru_size)
        self.ICMfeaturizer = IC_Features(self.num_feats)
        self.ICMForwardModel = IC_ForwardModel_Head(self.ICMfeaturizer.feature_size(), self.num_actions, self.ICMfeaturizer.feature_size())
        self.ICMBackwardModel = IC_InverseModel_Head(self.ICMfeaturizer.feature_size()*2, self.num_actions)        

    def icm_get_features(self, s):
        return self.ICMfeaturizer(s)

    def icm_get_forward_outp(self, phi, actions):
        return self.ICMForwardModel(phi, actions)

    def icm_get_inverse_outp(self, phi, actions):
        num = phi.size(0) - self.config.num_agents
        cat_phi = torch.cat((phi[:num], phi[self.config.num_agents:]), dim=1)
        
        logits = self.ICMBackwardModel(cat_phi)

        return logits

    def compute_intrinsic_reward(self, rollout):
        obs_shape = rollout.observations.size()[2:]
        num_steps, num_processes, _ = rollout.rewards.size()

        minibatch_size = rollout.observations[:-1].view(-1, *obs_shape).size(0)//self.config.icm_minibatches
        all_intr_reward = torch.zeros(rollout.rewards.view(-1, 1).shape, device=self.config.device, dtype=torch.float)

        minibatches = list(range(self.config.icm_minibatches))
        random.shuffle(minibatches)
        for i in minibatches:
          start=i*(minibatch_size)
          end=start+minibatch_size
          
          #compute intrinsic reward
          with torch.no_grad():
            phi = self.icm_get_features(rollout.observations.view(-1, *obs_shape)[start:end+self.config.num_agents])
          
            icm_obs_pred = self.icm_get_forward_outp(phi[:-1*self.config.num_agents], rollout.actions.view(-1, 1)[start:end])
            obs_diff = icm_obs_pred - phi[self.config.num_agents:]
            intr_reward = obs_diff.pow(2).sqrt().sum(dim=1) * self.config.icm_prediction_beta
            
            all_intr_reward[start:end] = intr_reward.view(-1, 1)
        
        rollout.rewards += all_intr_reward.view(num_steps, num_processes, 1)
        rollout.rewards = torch.clamp(rollout.rewards, min=-1.0, max=1.0)

    def update_icm(self, rollout, frame):
        obs_shape = rollout.observations.size()[2:]
        action_shape = rollout.actions.size()[-1]
        num_steps, num_processes, _ = rollout.rewards.size()

        total_forward_loss = 0.
        total_inverse_loss = 0.

        minibatch_size = rollout.observations[:-1].view(-1, *obs_shape).size(0)//self.config.icm_minibatches
        all_intr_reward = torch.zeros(rollout.rewards.view(-1, 1).shape, device=self.config.device, dtype=torch.float)

        minibatches = list(range(self.config.icm_minibatches))
        random.shuffle(minibatches)
        for i in minibatches:
          #forward model loss
          start=i*(minibatch_size)
          end=start+minibatch_size

          phi = self.icm_get_features(rollout.observations.view(-1, *obs_shape)[start:end+self.config.num_agents])
          tmp = rollout.observations[1,0]==rollout.observations.view(-1, *obs_shape)[self.config.num_agents]
          
          icm_obs_pred = self.icm_get_forward_outp(phi[:-1*self.config.num_agents], rollout.actions.view(-1, 1)[start:end])
          
          #forward model loss
          obs_diff = icm_obs_pred - phi[self.config.num_agents:]
          forward_model_loss = 0.5 * obs_diff.pow(2).sum(dim=1).mean()
          forward_model_loss *= float(icm_obs_pred.size(1)) #lenFeatures=288. Factored out to make hyperparams not depend on it.

          #inverse model loss
          icm_action_logits = self.icm_get_inverse_outp(phi, rollout.actions.view(-1, 1)[start:end])
          m = nn.CrossEntropyLoss()
          inverse_model_loss = m(icm_action_logits, rollout.actions.view(-1)[start:end])

          #total loss
          forward_loss = (self.config.icm_loss_beta*forward_model_loss)
          inverse_loss = ((1.-self.config.icm_loss_beta)*inverse_model_loss)
          loss = (forward_loss+inverse_loss)/float(self.config.icm_minibatches)
          loss /= self.config.icm_lambda

          self.optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
          self.optimizer.step()

          total_forward_loss += forward_loss.item()
          total_inverse_loss += inverse_loss.item()
          
        total_forward_loss /= float(self.config.icm_minibatches)
        total_forward_loss /= self.config.icm_lambda
        total_inverse_loss /= float(self.config.icm_minibatches)
        total_inverse_loss /= self.config.icm_lambda
        
        self.tb_writer.add_scalar('Loss/Forward Dynamics Loss', total_forward_loss, frame)
        self.tb_writer.add_scalar('Loss/Inverse Dynamics Loss', total_inverse_loss, frame)
        
        return total_forward_loss + total_inverse_loss

    def compute_loss(self, rollouts, next_value, frame):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        dynamics_loss = self.update_icm(rollouts, frame)
        self.compute_intrinsic_reward(rollouts)
        rollouts.compute_returns(next_value, self.config.GAMMA)

        values, action_log_probs, dist_entropy, states = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1),
            rollouts.states[0].view(-1, self.model.state_size),
            rollouts.masks[:-1].view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        pg_loss = action_loss + self.config.value_loss_weight * value_loss

        loss = pg_loss - self.config.entropy_loss_weight * dist_entropy

        self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Policy Loss', action_loss.item(), frame)
        self.tb_writer.add_scalar('Loss/Value Loss', value_loss.item(), frame)

        self.tb_writer.add_scalar('Policy/Entropy', dist_entropy.item(), frame)
        self.tb_writer.add_scalar('Policy/Value Estimate', values.detach().mean().item(), frame)

        self.tb_writer.add_scalar('Learning/Learning Rate', np.mean([param_group['lr'] for param_group in self.optimizer.param_groups]), frame)

        return loss, action_loss, value_loss, dist_entropy, dynamics_loss

    def save_w(self, best=False):
      if best:
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best', 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'best', 'optim.dump'))
        torch.save(self.ICMfeaturizer.state_dict(), os.path.join(self.log_dir, 'best', 'featurizer.dump'))
        torch.save(self.ICMBackwardModel.state_dict(), os.path.join(self.log_dir, 'best', 'backward.dump'))
        torch.save(self.ICMForwardModel.state_dict(), os.path.join(self.log_dir, 'best', 'forward.dump'))
      
      torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'saved_model', 'model.dump'))
      torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'optim.dump'))
      torch.save(self.ICMfeaturizer.state_dict(), os.path.join(self.log_dir, 'saved_model', 'featurizer.dump'))
      torch.save(self.ICMBackwardModel.state_dict(), os.path.join(self.log_dir, 'saved_model', 'backward.dump'))
      torch.save(self.ICMForwardModel.state_dict(), os.path.join(self.log_dir, 'saved_model', 'forward.dump'))

    def load_w(self, best=False):
      if best:
        fname_model = os.path.join(self.log_dir, 'best', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'best', 'optim.dump')
        fname_featurizer = os.path.join(self.log_dir, 'best', 'featurizer.dump')
        fname_backward = os.path.join(self.log_dir, 'best', 'backward.dump')
        fname_forward = os.path.join(self.log_dir, 'best', 'forward.dump')
      else:
        fname_model = os.path.join(self.log_dir, 'saved_model', 'model.dump')
        fname_optim = os.path.join(self.log_dir, 'saved_model', 'optim.dump')
        fname_featurizer = os.path.join(self.log_dir, 'saved_model', 'featurizer.dump')
        fname_backward = os.path.join(self.log_dir, 'saved_model', 'backward.dump')
        fname_forward = os.path.join(self.log_dir, 'saved_model', 'forward.dump')

      if os.path.isfile(fname_model):
        self.model.load_state_dict(torch.load(fname_model))

      if os.path.isfile(fname_optim):
        self.optimizer.load_state_dict(torch.load(fname_optim))

      if os.path.isfile(fname_featurizer):
        self.ICMfeaturizer.load_state_dict(torch.load(fname_featurizer))

      if os.path.isfile(fname_backward):
        self.ICMBackwardModel.load_state_dict(torch.load(fname_backward))

      if os.path.isfile(fname_forward):
        self.ICMForwardModel.load_state_dict(torch.load(fname_forward))
