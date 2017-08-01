
import pdb
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

fp = './data/mtl_15w_train.csv'

queue = tf.train.string_input_producer(["file0.csv"], num_epochs=1)
reader = tf.TextLineReader()
key, value = reader.read(queue)

rds = [['']] * 30   # all string
cols = tf.decode_csv(value, record_defaults=rds, field_delim='\t')


with tf.Session() as sess:
    
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    cnt = 0
    try:
        while not coord.should_stop():
            [fs] = sess.run([cols])
            cnt += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        print('cnt %d'%cnt)

    coord.request_stop()
    coord.join(threads)

