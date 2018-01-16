import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.saver = None

        self._build_network()

    # 인공신경망 모델을 구축하는 함수 
    # h_size : hidden size
    # l_rate : learning rate
    def _build_network(self, h_size = 46, l_rate=1e-3):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weights
            # 1번째 레이어
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            # 2번째 레이어
            W2 = tf.get_variable("W2", shape=[h_size, h_size*2], initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))

            # 3번째 레이어
            W3 = tf.get_variable("W3", shape=[h_size*2, h_size*3], initializer=tf.contrib.layers.xavier_initializer())
            layer3 = tf.nn.tanh(tf.matmul(layer2, W3))

			# 3번째 레이어
            W4 = tf.get_variable("W4", shape=[h_size*3, h_size*3], initializer=tf.contrib.layers.xavier_initializer())
            layer4 = tf.nn.tanh(tf.matmul(layer3, W4))

			# 3번째 레이어
            W5 = tf.get_variable("W5", shape=[h_size*3, h_size*4], initializer=tf.contrib.layers.xavier_initializer())
            layer5 = tf.nn.tanh(tf.matmul(layer4, W5))

			# 3번째 레이어
            W6 = tf.get_variable("W6", shape=[h_size*4, h_size*4], initializer=tf.contrib.layers.xavier_initializer())
            layer6 = tf.nn.tanh(tf.matmul(layer5, W6))

			# 3번째 레이어
            W7 = tf.get_variable("W7", shape=[h_size*4, h_size*3], initializer=tf.contrib.layers.xavier_initializer())
            layer7 = tf.nn.tanh(tf.matmul(layer6, W7))

			# 3번째 레이어
            W8 = tf.get_variable("W8", shape=[h_size*3, h_size*3], initializer=tf.contrib.layers.xavier_initializer())
            layer8 = tf.nn.tanh(tf.matmul(layer7, W8))

			# 3번째 레이어
            W9 = tf.get_variable("W9", shape=[h_size*3, h_size*2], initializer=tf.contrib.layers.xavier_initializer())
            layer9 = tf.nn.tanh(tf.matmul(layer8, W9))

			# 3번째 레이어
            W10 = tf.get_variable("W10", shape=[h_size*2, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer10 = tf.nn.tanh(tf.matmul(layer9, W10))

            # 4번째 레이어
            W11 = tf.get_variable("W11", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

            self._Qpred = tf.matmul(layer10, W11)

        # We need to define the parts of the network needed for learning a policy
        # 정답데이터 
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

        self.saver = tf.train.Saver({'W1' : W1, 'W2' : W2, 'W3' : W3, 'W4' : W4, 'W5' : W5, 'W6' : W6, 'W7' : W7, 'W8' : W8, 'W9' : W9, 'W10' : W10, 'W11' : W11})
        #END-------------------------------------------------------------------------

    # 훈련데이터를 예측하는 함수
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})


    # 학습데이터를 학습하는 함수
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
