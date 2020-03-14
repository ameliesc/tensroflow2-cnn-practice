import tensorflow_datasets as tfds
import tensorflow as tf

class DataLoader(object):  # lets use CIFAR dataset! make your own data loader
    def __init__(self, image_size, batch_size):

        self.image_size = image_size
        self.batch_size = batch_size

        # 80% train data, 10% validation data, 10% test data
        split_weights = (8, 1, 1)
        splits = tfds.Split.TRAIN.subsplit(weighted=split_weights)
        test_split, val_split = tfds.Split.TEST.subsplit(2)

        (self.test_data_raw, self.validation_data_raw, self.train_data_raw), self.metadata = tfds.load('cifar10', split=['test[:50]', 'train[90:]', 'train[:90]'], with_info=True, as_supervised=True)

        # fetches DatasetBuilder - returns tf Dataset or Tensor
        # (self.train_data_raw, self.validation_data_raw, self.test_data_raw), self.metadata = tfds.load(
        #    'cifar10', split=list(splits),
        #    with_info=True, as_supervised=True)

        # Get the number of train examples
        self.num_train_examples = self.metadata.splits['train'].num_examples * 90 / 100
        self.get_label_name = self.metadata.features['label'].int2str

        # Pre-process data
        self._prepare_data()
        self._prepare_batches()

    # Resize all images to image_size x image_size ## tod do: need to make this so it machtes cifar10 dataset
    def _prepare_data(self):
        #self.train_data = self.train_data_raw.map(self._augmentate_data)
        #self.validation_data = self.validation_data_raw.map(self._augmentate_data)
        #self.test_data = self.test_data_raw.map(self._augmentate_data)
        self.train_data = self.train_data_raw.map(self._preprocess_sample)
        self.validation_data = self.validation_data_raw.map(self._preprocess_sample)
        self.test_data = self.test_data_raw.map(self._preprocess_sample)

    def _augmentate_data(self, image, labels):
        image = tf.image.resize_with_crop_or_pad(
            image, 32 + 8, 32 + 8)
        image = tf.image.random_crop(image, [32, 32, 3])  # change to cariable, image height, width and channels
        image = tf.image.random_flip_left_right(image)
        return image, labels
    # Resize one image to image_size x image_size  and normalize data

    def _normalize(self, image):
        image = tf.image.per_image_standardization(image)
        return image

    def _preprocess_sample(self, image, label):
        image = tf.cast(image, tf.float32)
        image = self._normalize(image)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return image, label

    def _prepare_batches(self):
        self.train_batches = self.train_data.shuffle(1000).batch(self.batch_size)
        self.validation_batches = self.validation_data.batch(self.batch_size)
        self.test_batches = self.test_data.batch(self.batch_size)

    # Get defined number of  not processed images
    def get_random_raw_images(self, num_of_images):
        random_train_raw_data = self.train_data_raw.shuffle(1000)
