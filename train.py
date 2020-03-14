from model import MyLittleCNN
from prepare_data import DataLoader
import os
import tensorflow as tf

IMG_SIZE = 32
BATCH_SIZE = 32
SHUFFLE_SIZE = 1000
EPOCHS = 30
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

if not os.path.exists("training"):
    os.mkdir("training")
    
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_name = checkpoint_path.split('/')[-1]
checkpoint_dir = os.path.dirname(checkpoint_path)
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

data_loader = DataLoader(IMG_SIZE, BATCH_SIZE)

model = MyLittleCNN(optimizer=opt, loss=loss, input_shape=(None, 32, 32, 3), training=True)

history = model.fit_and_save_checkpoints(data_loader.train_batches, data_loader.validation_batches, epochs=EPOCHS, checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name)

test_loss, test_acc = model.evaluate(data_loader.test_batches, steps=20, verbose=2)
print("Model accuracy after training for %s epochs: %s" % (EPOCHS,test_acc))

