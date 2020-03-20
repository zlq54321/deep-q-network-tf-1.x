import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE = 0.01
EVALUATE_SAVE_PATH = './save_evaluate_net/'
EVALUATE_CKPT_FILE = './save_evaluate_net/evaluate_net.ckpt'


class NNModel:
    def __init__(self, input_size, hidden_sizes, output_size, output_activation=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.output_activation = output_activation
        self.graph = None
        self.tensors = []
        self.all_vars = None

    def inference(self, input_tensor):
        size_out = input_tensor.shape[1]  # 为了让循环体统一格式
        layer = input_tensor
        reg = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        reg = None

        for layer_idx, hidden_size in enumerate(self.hidden_sizes):
            with tf.variable_scope('layer' + str(layer_idx + 1)):
                size_in = size_out
                size_out = hidden_size
                weights = tf.get_variable('weights',
                                          [size_in, size_out],
                                          tf.float32,
                                          tf.truncated_normal_initializer(stddev=0.1))
                if reg is not None:
                    tf.add_to_collection('losses', reg(weights))

                if layer_idx == 0:
                    layer = tf.nn.relu(tf.matmul(input_tensor, weights))
                else:
                    layer = tf.nn.relu(tf.matmul(layer, weights))

        weights = tf.get_variable('weights',
                                  [size_out, self.output_size],
                                  tf.float32,
                                  tf.truncated_normal_initializer(stddev=0.1))
        output = tf.matmul(layer, weights)
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def train(self, predict, label, lr=LEARNING_RATE):
        # 损失函数
        loss = tf.reduce_mean(tf.square(predict - label))
        # loss = loss + tf.add_n(tf.get_collection('losses'))

        global_step = tf.Variable(0, trainable=False)
        # 滑动平均
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # 学习率衰减
        learning_rate = tf.train.exponential_decay(lr,
                                                   global_step,
                                                   decay_steps=500,
                                                   decay_rate=LEARNING_RATE_DECAY)

        # 反向训练
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        train_op = tf.group(train_step, variable_averages_op)

        return loss, train_step


# 经验回放 capacity:最多存储几条经验
class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQNAgent:
    def __init__(self, env, hidden_sizes=[16,], gamma=0.99, epsilon=0.001, replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)
        self.evaluate_net = self.build_network(input_size=observation_dim,
                                               hidden_sizes=hidden_sizes,
                                               output_size=self.action_n)
        self.target_net = self.build_network(input_size=observation_dim,
                                             hidden_sizes=hidden_sizes,
                                             output_size=self.action_n)

    def build_network(self, input_size, hidden_sizes, output_size, output_activation=None):

        return NNModel(input_size, hidden_sizes, output_size, output_activation)

    def learn(self, sars, done, sess_var, step):

        sess_e, saver_e, sess_t, saver_t = sess_var
        observation, action, reward, next_observation = sars

        self.replayer.store(observation, action, reward, next_observation, done)  # 存储经验
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)  # 经验回放

        # 用目标网络计算最大动作价值
        next_qs = sess_t.run(self.target_net.tensors['qs'],
                             feed_dict={self.target_net.tensors['observations']: next_observations})
        next_max_qs = next_qs.max(axis=-1)

        us = rewards + self.gamma * (1.0 - dones) * next_max_qs
        # 评估网络价值拟合us, 把评估网络价值的对应动作处改为u即可作为拟合标签
        evaluate_q_outputs = self.evaluate_net.tensors['qs']
        states = self.evaluate_net.tensors['observations']
        labels = sess_e.run(evaluate_q_outputs,
                            feed_dict={states: observations})  # 这里还不是labels
        labels[np.arange(us.shape[0]), actions] = us  # 对应动作处改为u即变为训练的标签

        loss = self.evaluate_net.tensors['loss']
        train_op = self.evaluate_net.tensors['train_op']
        u = self.evaluate_net.tensors['us_label']
        loss_val, _ = sess_e.run((loss, train_op),
                                 feed_dict={states: observations, u: labels})
        step += 1

        if done:
            # 存储评估网络参数
            saver_e.save(sess_e, EVALUATE_CKPT_FILE, global_step=step)
            # 读评估网络参数
            ckpt = tf.train.get_checkpoint_state(EVALUATE_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver_t.restore(sess_t, ckpt.model_checkpoint_path)

        return loss_val, step

    def decide(self, observation, sess_e):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        qs = sess_e.run(self.evaluate_net.tensors['qs'],
                        feed_dict={self.evaluate_net.tensors['observations']: observation.reshape(-1, 2)})

        return np.argmax(qs)


class DoubleDQNAgent(DQNAgent):
    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)
        next_eval_qs = self.evaluate_net.predict(next_observations)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1.0 - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())


def play_qlearning(episodes, env, agent, train=False, render=False):

    # 在图eval_g中调出评估网络结构
    eval_g = tf.Graph()
    with eval_g.as_default():
        observations = tf.placeholder(tf.float32, [None, agent.evaluate_net.input_size], name='observations')
        us_label = tf.placeholder(tf.float32, [None, agent.evaluate_net.output_size], name='us_label')
        qs = agent.evaluate_net.inference(observations)
        loss, train_op = agent.evaluate_net.train(qs, us_label)
        agent.evaluate_net.tensors = {'observations': observations,
                                      'us_label': us_label,
                                      'qs': qs,
                                      'loss': loss,
                                      'train_op': train_op}
        agent.evaluate_net.all_vars = tf.global_variables()
    agent.evaluate_net.graph = eval_g

    # 在图targ_g中调出目标网络结构
    targ_g = tf.Graph()
    with targ_g.as_default():
        observations2 = tf.placeholder(tf.float32, [None, agent.target_net.input_size], name='observations2')
        qs2 = agent.target_net.inference(observations2)
        agent.target_net.tensors = {'observations': observations2,
                                    'qs': qs2}
        agent.target_net.all_vars = tf.global_variables()
    agent.target_net.graph = targ_g

    # 建立评估网络会话，读取评估网络参数
    saver_e = tf.train.Saver(agent.evaluate_net.all_vars, max_to_keep=1)
    with tf.Session(graph=eval_g) as sess_e:
        tf.global_variables_initializer().run()
        # 从存储的参数继续训练
        ckpt = tf.train.get_checkpoint_state(EVALUATE_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver_e.restore(sess_e, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            step = int(step)
        else:
            # 没有参数，保持初始化参数, 并存储起来供目标网络读取
            saver_e.save(sess_e, EVALUATE_CKPT_FILE, global_step=0)
            step = 0

        # 建立目标网络会话
        saver_t = tf.train.Saver(agent.target_net.all_vars)
        with tf.Session(graph=agent.target_net.graph) as sess_t:
            # 读取评估网络参数
            ckpt = tf.train.get_checkpoint_state(EVALUATE_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver_t.restore(sess_t, ckpt.model_checkpoint_path)

            sess_var = (sess_e, saver_e, sess_t, saver_t)
            episode_rewards = []
            episode_losses = []

            for episode in range(episodes):

                observation = env.reset()
                episode_reward = 0
                episode_loss = 0
                last_step = step
                while True:

                    if render:
                        env.render()

                    action = agent.decide(observation, sess_e)
                    next_observation, reward, done, _ = env.step(action)
                    episode_reward += reward

                    if train:
                        sars = (observation, action, reward, next_observation)
                        loss_val, step = agent.learn(sars, done, sess_var, step)
                        episode_loss += loss_val
                    if done:
                        break

                    observation = next_observation

                print('Ep'+str(episode), 'steps:', step - last_step, 'reward:', episode_reward)
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss)

                if (episode + 1) % 10 == 0:
                    plt.close()
                    ax1 = plt.subplot(2, 1, 1)
                    ax2 = plt.subplot(2, 1, 2)
                    plt.sca(ax1)
                    plt.plot(episode_rewards, color='red')
                    plt.sca(ax2)
                    plt.plot(episode_losses, color='blue')
                    plt.draw()
                    plt.pause(1)

    return episode_rewards, episode_losses


def main(argv=None):
    np.random.seed(0)
    tf.random.set_random_seed(0)
    # 小车上山环境建立
    env = gym.make('MountainCar-v0')
    env.seed(0)

    # 深度Q学习智能体对象建立
    agent = DQNAgent(env, hidden_sizes=[64])

    episodes = 1000
    play_qlearning(episodes, env, agent, train=True)

    # 测试训练后的deep q， train=False
    # epsilon=0, 就是完全按照q决定的确定性策略decide action
    agent.epsilon = 0
    episode_rewards, _ = play_qlearning(100, env, agent)
    print('平均回合奖励 =  %.2f' % (np.mean(episode_rewards)))

    env.close()


if __name__ == '__main__':
    tf.app.run()
