from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


max_steps = 1000
learning_rate = 0.001
dropout =0.9

data_dir = "MNIST_data/" #数据路径
log_dir = 'tmp/mnist_with_summaries' #日志路径

sess = tf.InteractiveSession()

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

def variable_summaries(param):
    '''
    summary汇总
    :param var:
    :return:
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(param) #计算均值
        tf.summary.scalar('mean',mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max', tf.reduce_max(param))
        tf.summary.scalar('min', tf.reduce_min(param))
        tf.summary.histogram('histogram',param)



#加载原始数据
mnist = input_data.read_data_sets(data_dir, one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x-inpu')  # x输入，28*28的
    y_ = tf.placeholder(tf.float32, [None, 10],name='y-input')  # label的输出

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input_image',image_shaped_input,10) #输入图片

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

saver = tf.train.Saver()
for i in range(max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+"/model.ckpt", i)#保存模型
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()

