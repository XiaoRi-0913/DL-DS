import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.decomposition import PCA
from scipy import stats
from pandas.core.frame import DataFrame

# 定义命令行参数
tf.app.flags.DEFINE_integer('is_train', 1, "指定是否是训练模型，还是拿数据去预测")
# 简化变量名
FLAGS = tf.app.flags.FLAGS


def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))  # 正态分布的标准差是0.01


def convert_frameToSet(inputData):
    return tf.data.Dataset.from_tensor_slices(dict(inputData))


def create_ds_model(x):
    y_predict = 0
    # 第一个卷积层
    with tf.variable_scope("conv1"):
        # 卷积层
        input_x = tf.reshape(x, shape=[-1, 13, 13, 1])
        # 定义卷积核 偏置
        conv1_weights = create_weights(shape=[5, 5, 1, 32])

        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias
        # 输出 conv1_x = [batch_size,13,13,32] 最后一个32是通道
        # 激活函数
        relu1_x = tf.nn.relu(conv1_x)
        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # pool1_x = [batch_size,7,7,32]
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
        # pool2_x = [batch_size,4,4,64]
    # 全连接层
    with tf.variable_scope("full_connection"):
        # '''
        # [batch_size,4,4,64] 重构形状 [batch_size,4*4*64] * [4*4*64, 10]
        # 降维 [batch_size,10]
        # '''
        x_fc = tf.reshape(pool2_x, shape=[-1, 4 * 4 * 64])
        weights_fc = create_weights(shape=[4 * 4 * 64, 10])
        bias_fc = create_weights(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc
    return y_predict


def operation_data(X, Y):
    itrait = 0  # first trait analyzed
    X_train, X_test, y_train, y_test = train_test_split(X, Y[itrait], test_size=0.2)
    X = np.concatenate((X_train, X_test))  # 把训练集和测试集拼接
    pca = PCA(n_components=2)  # 减少到两个特征
    p = pca.fit(X).fit_transform(X)
    Ntrain = X_train.shape[0]
    # OPTIONAL: SNP preselection according to a simple GWAS
    pvals = []
    for i in range(X_train.shape[1]):
        b, intercept, r_value, p_value, std_err = stats.linregress(X_train[i], y_train)  # 两组测量值计算线性least-squares回归
        pvals.append(-np.log10(p_value))
    pvals = np.array(pvals)
    ## b是斜率，intercept是截距， p值， 标准误差
    # select N_best most associated SNPs
    # N_best = X_train.shape[1] #all SNPs
    N_best = 100
    snp_list = pvals.argsort()[-N_best:]

    # or select by min_P_value
    min_P_value = 2  # P = 0.01
    snp_list = np.nonzero(pvals > min_P_value)
    # finally slice X
    X_train = X_train[X_train.columns[snp_list]]
    X_test = X_test[X_test.columns[snp_list]]
    return X_train, X_test, y_train, y_test


def tensor_ds():
    # 1.准备数据
    X = pd.read_csv('DATA/wheat.X', header=None, sep='\s+')
    Y = pd.read_csv('DATA/wheat.Y', header=None, sep='\s+')
    X_train, X_test, y_train, y_test = operation_data(X, Y)
    X_train = convert_frameToSet(X_train)
    X_test = convert_frameToSet(X_test)
    y_train = convert_frameToSet(DataFrame(y_train))
    y_test = convert_frameToSet(DataFrame(y_test))
    with tf.variable_scope("ds_data"):
        X_train = tf.placeholder(tf.float32, [None, 169])
        y_true = tf.placeholder(tf.int32, [None, 10])
    y_predict = create_ds_model(X_train)
    # 3、softmax回归以及交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):
        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        print("--------------------------------------------------")
        print(y_true)
        print(y_predict)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_predict))

    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)  # 优化器
        # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # （2）收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # （3）合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # 加载模型
        # if os.path.exists("./tmp/modelckpt/checkpoint"):
        #     saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:
            # 循环步数去训练
            for i in range(100):
                # 获取数据，实时提供
                # 每步提供50个样本训练
                train_x = X_train[:, 50 * i + 1]
                train_y = y_train[50 * i + 1]
                # 运行训练op
                sess.run(train_op, feed_dict={X_train: train_x, y_true: train_y})
                print("训练第%d步的准确率为：%f, 损失为：%f " % (i + 1,
                                                   sess.run(accuracy, feed_dict={X_train: train_x, y_true: train_y}),
                                                   sess.run(loss, feed_dict={X_train: train_x, y_true: train_y})
                                                   )
                      )

                # 运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={X_train: train_x, y_true: train_y})
                file_writer.add_summary(summary, i)
                # if i % 100 == 0:
                #     saver.save(sess, "./tmp/modelckpt/fc_nn_model")

        else:
            # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                test_x = X_test[:, 50 * i + 1]
                test_y = y_test[50 * i + 1]
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={X_test: test_x, y_true: test_y}), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={X_test: test_x, y_true: test_y}), 1).eval()
                )
                      )

    return None


def testDataSet():
    X = pd.read_csv('DATA/wheat.X', header=None, sep='\s+')
  #  mnist = input_data.read_data_sets("../mnist_data", one_hot=True)
    ds = tf.data.Dataset.from_tensor_slices(dict(X))
    print(X.head())
    print(ds)
    return None


if __name__ == '__main__':
    tensor_ds()
