#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow.compat.v1 as tf
import time
import random

# 让 TF2 像 TF1 一样运行
tf.disable_v2_behavior()

print("TensorFlow version:", tf.__version__)  # 建议 >=2.3


# PhysicsInformedNN class
class PhysicsInformedNN:
    def __init__(self, t, z, theta, layers_psi, layers_theta, layers_K):
        self.t = t
        self.z = z
        self.theta = theta

        self.layers_psi = layers_psi
        self.layers_theta = layers_theta
        self.layers_K = layers_K

        # initialize NNs
        self.weights_psi, self.biases_psi = self.initialize_NN(layers_psi)
        self.weights_theta, self.biases_theta = self.initialize_NN(layers_theta)
        self.weights_K, self.biases_K = self.initialize_NN(layers_K)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # tf placeholder
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])

        # prediction from PINNs
        self.theta_pred, self.psi_pred, self.K_pred, \
        self.f_pred, self.theta_t_pred, self.psi_z_pred, \
        self.psi_zz_pred, self.K_z_pred = self.net(self.t_tf, self.z_tf)

        # loss: 数据项 + PDE项（降低权重避免数值爆炸）
        self.loss = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred)) \
                  + 1e-3 * tf.reduce_mean(tf.square(self.f_pred))

        # optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Xavier 初始化
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=xavier_stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # psi 网络
    def net_psi(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]; b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]; b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        # 稳定化：避免 exp 溢出
        psi = -tf.exp(tf.clip_by_value(Y, -20, 20))
        return psi

    # theta 网络
    def net_theta(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]; b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]; b = biases[-1]
        theta = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        return theta

    # K 网络
    def net_K(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]; b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]; b = biases[-1]
        K = tf.exp(tf.add(tf.matmul(H, W), b))
        return K

    # 总网络
    def net(self, t, z):
        X = tf.concat([t, z], 1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi)

        # 稳定化 log(-psi)
        log_h = tf.math.log(tf.maximum(-psi, 1e-6))
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]

        f = theta_t - K_z * psi_z - K * psi_zz - K_z
        return theta, psi, K, f, theta_t, psi_z, psi_zz, K_z

    # 训练
    def train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.z_tf: self.z, self.theta_tf: self.theta}

        start_time = time.time()
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()

    def predict(self, t_star, z_star):
        tf_dict = {self.t_tf: t_star, self.z_tf: z_star}
        theta_star = self.sess.run(self.theta_pred, tf_dict)
        psi_star = self.sess.run(self.psi_pred, tf_dict)
        K_star = self.sess.run(self.K_pred, tf_dict)
        return theta_star, psi_star, K_star


# =======================
# 运行 main_loop 示例
# =======================
def main_loop():
    tf.reset_default_graph()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    # 随机数据测试
    t = np.random.rand(100, 1)
    z = np.random.rand(100, 1)
    theta = np.random.rand(100, 1)

    layers_psi = [2, 20, 20, 1]
    layers_theta = [1, 20, 1]
    layers_K = [1, 20, 1]

    model = PhysicsInformedNN(t, z, theta, layers_psi, layers_theta, layers_K)
    model.train(200)  # 迭代少一点方便测试
    theta_pred, psi_pred, K_pred = model.predict(t, z)
    print("theta_pred shape:", theta_pred.shape)


if __name__ == "__main__":
    main_loop()
