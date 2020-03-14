import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras import layers
import tensorflow_datasets as tfds

### Todo for making model mor generalizeable :
#-save history when fitting, create plot function for visualizing history
#-generalize fit function to add other callback functions as well
#-generalize model independent of dataset
#-make extra function wher model is defined, build and compiled so that it can be overrun or redefined when necessary 
class MyLittleCNN(tf.keras.Model):
    """ CNN class build on keras base model. CNN is defined in init function, and build and compiled at the same time. Currently build for Cifar10 dataset 
    Other properties:
    : call: for inference
    : fit_and_save_checkpoints: just like keras fit function but with checkpoint call back added
    : load_from_checkpoint: load model from checkpoint 
    : plot_results: plots input image and class along with predicted class 
    : save_frozen_graph: saves tensorflow frozen graph with tensorflow 1.x compatibility
    : load_frozen_graph: loads tensorflow forzen graph with tensorflow 1.x compatibility
    """ 
     
    def __init__(self, optimizer, loss, input_shape, training=False ):
        """
        CNN graph is defined, build and compiled here.
        Input:
        : optimizer: tensorflow optimizer function
        : loss: tensorflow loss function
        : input_shape: shape of input image (current cnn is build for cifar10 dataset)
        : training: boolian - defines wether model should be trained or not 
        """
        super(MyLittleCNN,self).__init__()
        self.trainable = training
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape))
        
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64, activation='relu')
        if self.trainable:
           self.dropout = layers.Dropout(rate = 0.5)
        self.output_layer = layers.Dense(10, activation='softmax')
        
        self.build( input_shape=input_shape) 
        self.summary()
        self.compile(optimizer=optimizer, loss = loss, metrics = ['accuracy'])
                         
    def call(self, inputs, training = False ):
        
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
       
        if self.trainable:
            x = self.dropout(x, training=training)
        return self.output_layer(x)

     

    def fit_and_save_checkpoints(self,train_data_batches,validation_data_batches,epochs,checkpoint_dir, checkpoint_name = ""):
        """ Calls keras model fit function and saves checkpoints when an improvement in validation accruacy is found.
        Input: 
         :train_data_batches: traindata from batches from tensorflow data generator 
         :validation_data_batches: validationdata batches from tensorflow data generator 
         :epochs: Integer - number of epchs 
         :checkpoint_dir: String - path to checkpoint directory 
         :checkpoint_name: String - checkpoint name 
        """
        ### to do -> transform into generalized fit function where one can add more than just the checkpoint_callback 
        checkpoint_callback =tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir,checkpoint_name), monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=False,
            save_frequency=1)
        return self.fit(train_data_batches, epochs=epochs, validation_data = validation_data_batches, callbacks = [checkpoint_callback])

    def load_from_checkpoint(self, checkpoint_dir, checkpoint_name = ""):
        
        latest = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, checkpoint_name))
        self.load_weights(latest)
        

    def plot_results(self, input_image,label, prediction):
        """Visualizes input image, true label and its predicted label"""
        class_name_pred = self.map_labels(prediction)
        class_name_true = self.map_labels(prediction)
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(input_image,cmpa=plt.cm.binary)
        plt.xlabel("True class:{}".format(class_name_true))
        plt.subplot(1,2,2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(input_image, cmap=plt.cm.binary)
        plt.xlabel("Predicted class:{}".format(class_name_pred))
        plt.show()
        

    
    def map_labels(self, prediction):
        """maps prediction integer to class name
        Input: 
         :prediction: integer [0-9]
        Output: 
         :class_name: string 
        """
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        
        return class_names[prediction]

    def evaluate_from_checkpoint(self,checkpoint_dir, checkpoint_name = ""):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.load_weights(latest)
        test_loss, test_acc = self.evaluate(data_loader.test_batches,steps = 20, verbose=2)
        return test_acc

    def save_frozen_graph(): # for tensorflow1.x compatibility, depreciated in 2.x
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: self(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(self.inputs[0].shape, self.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)

    def _wrap_frozen_graph(self,graph_def, inputs, outputs, print_graph=False):
        def imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph

            print("-" * 50)
            print("Frozen model layers: ")
            layers = [op.name for op in import_graph.get_operations()]
            if print_graph == True:
                for layer in layers:
                    print(layer)
                    print("-" * 50)

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))

    def load_frozen_graph(self): # for tensorflow1 compability
        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = self._wrap_frozen_graph(graph_def=graph_def,
                    inputs=["x:0"],
                    outputs=["Identity:0"],
                    print_graph=True)



