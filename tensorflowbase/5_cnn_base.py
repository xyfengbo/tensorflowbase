from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



def weight_variable(shape):
    '''
    初始化权重
    :param shape:
    :return:
    '''
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    初始化偏置
    :param shape:
    :return:
    '''
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    '''
    卷积函数，输入和输出一样大
    :param x:
    :param W:[5,5,3,32]表示5*5的3通道，32个卷积核
    strides=[1,1,1,1]，表示步长为1,第一个和最后一个必须是1，中间两个数分别代表了水平滑动和垂直滑动步长值。
    padding='SAME' 表示卷积的输出和输入大小一致
    :return:
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    '''
    2X2的最大池化函数
    :param x:
    ksize：取一个四维向量，一般是[1, height, width, 1]，
    因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    :return:
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def cnn_main():
    '''
    cnn基础测试
    :return:
    '''
    #定义输入和输出
    x = tf.placeholder(tf.float32,[None,784])#x输入，28*28的
    y_= tf.placeholder(tf.float32,[None,10]) #label的输出
    x_image = tf.reshape(x,[-1,28,28,1]) #image变形，只有1个通道，变成28*28*1

    #定义第1次卷积
    conv_kernel1 =[5,5,1,32]#第一次卷积核大小是5*5*1，一个放32个卷积核
    W_conv1 = weight_variable(conv_kernel1)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #第一次卷积，图片28->28,大小不变，padding='SAME'
    h_pool1 = max_pool_2x2(h_conv1)#第一次池化，图片28->14

    #定义第2次卷积
    conv_kernel2 =[5,5,32,64]#第2次卷积核5*5*32，个数是64
    W_conv2 = weight_variable(conv_kernel2)
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#第二次卷积，图片14->14,大小不变，padding='SAME'
    h_pool2 = max_pool_2x2(h_conv2)#第二次池化，图片14->7

    #定义全连接层，1024个连接
    fc1_param = [7*7*64,1024]
    W_fc1 = weight_variable(fc1_param)
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    #定义Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #将drop层连接softmax
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    #运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        print('训练集合大小：',mnist.train.images.shape)  # 输出测试集大小
        print('测试集合大小：',mnist.test.images.shape)  # 输出测试集大小
        for i in range(10000):
            batch = mnist.train.next_batch(128)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy,feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        test_accuracy = sess.run(accuracy,feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("test accuracy %g" % test_accuracy)

if __name__ == '__main__':
    cnn_main()