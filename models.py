### train CNN script from standford course 

import os
import time
import tensorflow as tf
from tensorflow.keras import Model, datasets, layers, models
import utils

def get_data(train = False, test = False):
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
       
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    if train:
        return train_images, train_labels
    elif test:
        return test_images,test_labels
    else:
        return None ## throw error message here 


class ConvNet(Model):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = True


       ##  return generator/ iterator or whatever 
    def build(self):
       
       self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
       self.maxpool1 = layers.MaxPooling2D((2, 2)))
       self.conv2 = layers.Conv2D(64, (3, 3), activation='relu'))
       self.maxpool2 = model.add(layers.MaxPooling2D((2, 2)))
       self.conv3 = Conv2D(64, (3, 3), activation='relu'))
       self.flatten = layers.Flatten())
       self.fc1 = layers.Dense(64, activation='relu'))
       self.out = model.add(layers.Dense(10))

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1,28,28,1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.droput(x, training = is_training)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

    
      
    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        compute mean cross entropy, softmax is applied internally
        
        '''
        # add different loss options late
 
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        return tf.reduce_mean(entropy, name='loss')
    
    def optimizer(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        ## add different options later 
        self.optimizer =  tf.optimizers.Adam(self.lr)


def train():
    convnet = ConvNet()
   

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_mnist')
        writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=30)
