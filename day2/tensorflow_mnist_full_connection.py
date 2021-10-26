from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# modify by wong
def full_connection():
    """
    全连接层对手写数字识别

    2.构建模型
    3.构造损失函数
    4.优化损失
    :return:
    """
    # 1.准备数据
    mnist = input_data.read_data_sets("../mnist_data", one_hot=True)
    # 训练值
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    # 目标值
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, 10))
    # 2.构建模型
    # 权重 正态分布的随机值
    weights = tf.Variable(initial_value=tf.random_normal(shape=(784, 10)))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x, weights) + bias
    # 损失函数
    # 交叉熵+softmax
    error = tf.reduce_min(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 优化损失 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    # 初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        image, label = mnist.train.next_batch(100)
        print("训练之前损失为%f" % sess.run(error, feed_dict={x: image, y_true: label}))
        for i in range(100):
            _, loss = sess.run([optimizer, error], feed_dict={x: image, y_true: label})
            print("第%d次训练，损失值为%f" % (i + 1, loss))

    # 准确率计算
        # 比较输出的结果最大值和真实值的最大值所在的位置 一致返回1 否则返回0 tf.argmax
        # 求平均
    return None


if __name__ == '__main__':
    full_connection()
