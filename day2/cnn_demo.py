from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 定义命令行参数
tf.app.flags.DEFINE_integer('is_train', 1, "指定是否是训练模型，还是拿数据去预测")
# 简化变量名
FLAGS = tf.app.flags.FLAGS


def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))  # 正态分布的标准差是0.01


def create_model(x):
    """
    构建CNN
    :param x:
    :return:
    """
    y_predict = 0

    # 第一个卷积层
    with tf.variable_scope("conv1"):  # 设置单独的空间
        # 卷积层
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 定义卷积核 偏置
        conv1_weights = create_weights(shape=[5, 5, 1, 32])
        conv1_bias = create_weights(shape=[32])

        conv1_x = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME")+ conv1_bias
        # 输出 conv1_x = [batch_size,14,14,32] 最后一个32是通道
        # 激活函数
        relu1_x = tf.nn.relu(conv1_x)
        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        # pool1_x = [batch_size,14,14,32]
    # 第二个卷积层
    with tf.variable_scope("conv2"):
        # 定义卷积核 偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_bias
        # 输出 conv2_x  = [batch_size,14,14,64] 最后一个32是通道
        # 激活函数
        relu2_x = tf.nn.relu(conv2_x)
        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # pool2_x = [batch_size,7,7,64]
    # 全连接层
    with tf.variable_scope("full_connection"):
        # '''
        # [batch_size,7,7,64] 重构形状 [batch_size,7*7*64] * [7*7*64, 10]
        # 降维 [batch_size,10]
        # '''
        x_fc = tf.reshape(pool2_x, shape=[-1, 7 * 7 * 64])
        weights_fc = create_weights(shape=[7 * 7 * 64, 10])
        bias_fc = create_weights(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def cnn_mnist():
    # 1.准备数据
    mnist = input_data.read_data_sets("../mnist_data", one_hot=True)
    # 图片的数据，32*32 像素单通道 784 None先不确定 表示图片的张数
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    y_predict = create_model(x)
    # 3.构造损失函数 分类任务，这里用交叉信息熵
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 5. 计算准确率
    bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))

    # 创建saver 对象
    saver = tf.train.Saver()
    # 初始化变量
    init = tf.global_variables_initializer()

    init_loss = float('inf')

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        if FLAGS.is_train == 1:
            image, label = mnist.train.next_batch(100)
            print("训练之前损失为%f" % sess.run(error, feed_dict={x: image, y_true: label}))

            # 开始训练
            for i in range(100):
                _, loss = sess.run([optimizer, error], feed_dict={x: image, y_true: label})
                if loss < init_loss:
                    init_loss = loss
                    saver.save(sess, '../tmp/model/mnist.ckpt')
                if (i + 1) % 10 == 0:
                    print("第%d次训练,损失为%f" % (i + 1, loss))
        else:
            # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                mnist_x, mnist_y = mnist.test.next_batch(1)
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                )
                      )
    return None


if __name__ == '__main__':
    cnn_mnist()