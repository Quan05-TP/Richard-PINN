# -*- coding: utf-8 -*-
"""
Richards 方程 PINN 模板
TensorFlow 2.x 运行在 compat.v1 模式
"""

import tensorflow as tf
import numpy as np

# 让 TF2 像 TF1 一样运行
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()



# van Genuchten 参数
theta_r = 0.078   # 残余含水量
theta_s = 0.43    # 饱和含水量
alpha = 0.036     # cm^-1
n = 1.56
m = 1 - 1/n
Ks = 0.00944      # cm/s


def theta(h):
    """含水量函数"""
    Se = (1 + (alpha * tf.abs(h))**n)**(-m)
    return theta_r + (theta_s - theta_r) * Se


def K(h):
    """导水率函数"""
    Se = (1 + (alpha * tf.abs(h))**n)**(-m)
    return Ks * tf.sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2


# 定义 PINN 网络
def neural_net(X, layers):
    """全连接前馈网络"""
    num_layers = len(layers)
    H = X
    for l in range(0, num_layers-2):
        W = tf.Variable(tf.random.normal([layers[l], layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    # 输出层
    W = tf.Variable(tf.random.normal([layers[-2], layers[-1]], dtype=tf.float32), dtype=tf.float32)
    b = tf.Variable(tf.zeros([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
    Y = tf.add(tf.matmul(H, W), b)
    return Y


# 数据
N_f = 2000   # PDE collocation points
N_u = 100    # 边界点

z_f = np.random.rand(N_f, 1)  # 空间点
t_f = np.random.rand(N_f, 1)  # 时间点

z_u = np.vstack([np.zeros((N_u, 1)), np.ones((N_u, 1))])  # 上下边界 z=0,1
t_u = np.random.rand(2*N_u, 1)
h_u = np.zeros((2*N_u, 1))  # 假设边界条件 h=0 (可改)

# 占位符
X_f = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])  # (z,t)
X_u = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
H_u = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 网络输出 h(z,t)
layers_dim = [2, 50, 50, 50, 1]
h_pred_f = neural_net(X_f, layers_dim)
h_pred_u = neural_net(X_u, layers_dim)

# 自动微分
grads_f = tf.gradients(h_pred_f, X_f)[0]
h_z = grads_f[:, 0:1]
h_t = grads_f[:, 1:2]
h_zz = tf.gradients(h_z, X_f)[0][:, 0:1]

# Richards 方程残差
theta_t = tf.gradients(theta(h_pred_f), X_f)[0][:, 1:2]
K_h = K(h_pred_f)
term = tf.gradients(K_h*(h_z+1), X_f)[0][:, 0:1]

f_pred = theta_t - term


# 损失函数
loss_pde = tf.reduce_mean(tf.square(f_pred))
loss_bc = tf.reduce_mean(tf.square(h_pred_u - H_u))
loss = loss_pde + loss_bc

# 优化器
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss)


# 训练

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(2000):
        X_f_train = np.hstack([z_f, t_f])
        X_u_train = np.hstack([z_u, t_u])
        feed_dict = {X_f: X_f_train, X_u: X_u_train, H_u: h_u}
        _, l = sess.run([train_op, loss], feed_dict=feed_dict)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss={l:.6f}")

    # 测试
    z_test = np.linspace(0, 1, 5)[:, None]
    t_test = np.zeros((5, 1))
    X_test = np.hstack([z_test, t_test])
    h_pred_test = sess.run(h_pred_u, feed_dict

    ={X_u: X_test})
    print("测试点预测 h(z,0)：", h_pred_test.flatten())
