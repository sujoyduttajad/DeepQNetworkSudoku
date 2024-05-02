import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

tf_v1.reset_default_graph()
tf_v1.disable_v2_behavior()

class CustomDeepQNetwork:
    def __init__(self, num_actions, num_features, learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=300, memory_size=500, batch_size=32,
                 e_greedy_increment=None, output_graph=False):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, num_features * 2 + 2))

        self._build_network()

        t_params = tf_v1.get_collection(tf_v1.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf_v1.get_collection(tf_v1.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")

        with tf_v1.variable_scope('hard_replacement'):
            self.target_replace_op = [tf_v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf_v1.Session()

        if output_graph:
            tf_v1.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf_v1.global_variables_initializer())
        self.cost_history = []

    def _build_network(self):
        self.state = tf_v1.placeholder(tf_v1.float32, [None, self.num_features], name='state')
        self.next_state = tf_v1.placeholder(tf_v1.float32, [None, self.num_features], name='next_state')
        self.reward = tf_v1.placeholder(tf_v1.float32, [None, ], name='reward')
        self.action = tf_v1.placeholder(tf_v1.int32, [None, ], name='action')

        w_initializer, b_initializer = tf_v1.random_normal_initializer(0., 0.3), tf_v1.constant_initializer(0.1)

        with tf_v1.variable_scope('evaluation_net'):
            e1 = tf_v1.layers.dense(self.state, 20, tf_v1.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e1')
            self.q_eval = tf_v1.layers.dense(e1, self.num_actions, kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer, name='q')

        with tf_v1.variable_scope('target_net'):
            t1 = tf_v1.layers.dense(self.next_state, 20, tf_v1.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t1')
            self.q_next = tf_v1.layers.dense(t1, self.num_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='t2')

        with tf_v1.variable_scope('q_target'):
            q_target = self.reward + self.reward_decay * tf_v1.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf_v1.stop_gradient(q_target)

        with tf_v1.variable_scope('q_eval'):
            a_indices = tf_v1.stack([tf_v1.range(tf_v1.shape(self.action)[0], dtype=tf_v1.int32), self.action], axis=1)
            self.q_eval_wrt_a = tf_v1.gather_nd(params=self.q_eval, indices=a_indices)

        with tf_v1.variable_scope('loss'):
            self.loss = tf_v1.reduce_mean(tf_v1.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf_v1.variable_scope('train'):
            self.train_op = tf_v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store_transition(self, state, action, reward, next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, [action, reward], next_state))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.num_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.state: batch_memory[:, :self.num_features],
                self.action: batch_memory[:, self.num_features],
                self.reward: batch_memory[:, self.num_features + 1],
                self.next_state: batch_memory[:, -self.num_features:],
            }
        )

        self.cost_history.append(cost)

        self.epsilon = self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
