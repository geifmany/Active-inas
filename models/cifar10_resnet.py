from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras import optimizers
import keras.backend as K

import numpy as np
from keras import regularizers
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


class Cifar10Resnet:
    def __init__(self, train_idx, pool_idx, depth_mult=1, structure=[3, 4, 6, 3]):
        '''

        :param train_idx: an array of indices for the current training set
        :param pool_idx: an array of indices for the current test set
        :param depth_mult: depth multiplier (not in use in our paper)
        :param structure: the architecture denoted as a list where each element is the number of blocks in the
            corresponding stack. For example A(B,2,4)=[2,2,2,2]
        '''
        self.num_classes = 10
        self.accuracy = 0
        self.search_results = {}
        self.weight_decay = 5e-4
        self.x_shape = [32, 32, 3]
        self.depth_multiplier = depth_mult
        self.structure = structure
        self.dr1 = K.variable(value=0)

        self.model = self._build_model(blocks=self.structure)

        self.train_idx = train_idx
        self.pool_idx = pool_idx
        self._load_data()
        self.sample_size = (train_idx.shape[0])
        self.search_res = {0: 0}

    def resnet_layer(self, inputs,
                     num_filters=32,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):

        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(self.weight_decay))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def depth_mult(self, original_depth):
        if self.depth_multiplier != 1:
            new_depth = int(original_depth * self.depth_multiplier + 1)
            new_depth = max(new_depth, 1)
        else:
            new_depth = original_depth
        return new_depth

    def _build_model(self, input_shape=[32, 32, 3], blocks=[3, 4, 6, 3], num_classes=10):
        '''
        Build a resnet model based on the architecture in the param blocks.
        :param input_shape: the size of the input images (3 dim)
        :param blocks: The architecture, where the first element represents the number of blocks in stack 1 etc.
        :param num_classes: number of lasses for the classification layer
        :return: a keras model
        '''

        # Start model definition.
        num_filters = (self.depth_mult(64))
        num_res_blocks = len(blocks)
        inputs = Input(shape=input_shape)
        # initial block
        x = self.resnet_layer(inputs=inputs, num_filters=num_filters)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        # Nested for loops for stacks and blocks
        for stack in range(num_res_blocks):
            for res_block in range(blocks[stack]):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters,
                                      strides=strides)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters,
                                      activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack add 1X1 to fix dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=True)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # classification block
        final_res = int(32 / (2 ** num_res_blocks))
        x = AveragePooling2D(pool_size=final_res)(x)
        y = Flatten()(x)

        y = Lambda(lambda x: K.dropout(x, level=self.dr1))(y)  # this is a dropout that applied manually for MC-dropout

        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        model = Model(inputs=inputs, outputs=outputs)
        # the embedding is defined for the coreset method
        self.embeding = Model(inputs=inputs, outputs=y)

        return model

    def normalize(self, X_train, X_test):
        '''
        This function normalize inputs for zero mean and unit variance

        Args:
            X_train: np array of train samples, axis 0 is samples.
            X_test: np array of test/validation samples, axis 0 is samples.

        Returns:
            A tuple (X_train, X_test), Normalized version of the data.

        '''
        self.mean = np.mean(X_train, axis=(0, 1, 2, 3))
        self.std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - self.mean) / (self.std + 1e-7)
        X_test = (X_test - self.mean) / (self.std + 1e-7)
        return X_train, X_test

    def normalize_production(self, x):
        '''
        This function is used to normalize instances in production according to saved training set statistics

        Input: X - a training set
        Output X - a normalized training set according to normalization constants.

        '''

        return (x - self.mean) / (self.std + 1e-7)

    def predict(self, x, batch_size=50):
        return self.model.predict(self.normalize_production(x), batch_size)

    def softmax_response(self, batch_size=200):
        '''
        Calculate the softmax response for the pool
        :param batch_size: batch size for model inference
        :return: array of softmax response for each point in the pool
        '''

        pred = self.model.predict(self.x_train[self.pool_idx], batch_size)
        soft_max = np.max(pred, 1)
        return soft_max

    def coreset_mat(self, batch_size=200):
        '''
        This function calculate the distance matrices for the coreset querying based on the embedding layer
        :param batch_size: batch size for model's inference
        :return: [mat_a, mat_b] where mat_a is distance matrix from training set to the pool, mat_b is a distance matrix
            from all points in the pool the the points in the pool (main diag is zero).
        '''
        train_emb = self.embeding.predict(self.x_train[self.train_idx], batch_size)
        pool_emb = self.embeding.predict(self.x_train[self.pool_idx], batch_size)
        return euclidean_distances(train_emb, pool_emb), euclidean_distances(pool_emb, pool_emb)

    def mc_dropout(self, batch_size=200):
        pred = self.model.predict(self.x_train[self.pool_idx], batch_size)
        cls = np.argmax(pred, 1)

        pred = []

        K.set_value(self.dr1, 0.5)

        for i in range(100):
            pred.append(self.model.predict(self.x_train[self.pool_idx], batch_size))
        K.set_value(self.dr1, 0)

        pred = np.array(pred)
        pred = np.var(pred, 0)
        confidence = -1 * pred[np.arange(self.pool_idx.shape[0]), cls]

        return confidence

    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train, self.x_test = self.normalize(x_train, x_test)

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

    def train(self, max_epoches=200, validation=True):
        '''

        :param max_epoches: the number of epochs to train
        :param validation: IF true, use the original validation set, If false, random split of 20% for validation
            as (used in iNAS).
        '''
        x_train = self.x_train[self.train_idx]
        y_train = self.y_train[self.train_idx]
        if validation:
            x_test = self.x_test
            y_test = self.y_test
            verbose = 2
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                                y_train,
                                                                test_size=0.2,
                                                                random_state=1)
            verbose = 0
        # training parameters
        batch_size = 128
        learning_rate = 0.1

        def lr_schedule(epoch):
            """Learning Rate Schedule
            Learning rate is scheduled to be reduced after 100 and 150 epochs.
            Called automatically every epoch as part of callbacks during training.
            # Arguments
                epoch (int): The number of epochs
            # Returns
                lr (float32): learning rate
            """
            lr = 0.1
            if epoch > 100:
                lr = 0.01
            if epoch > 150:
                lr = 0.001
            return lr

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

        train_history = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                              batch_size=batch_size),
                                                 steps_per_epoch=50000 // batch_size,
                                                 epochs=max_epoches,
                                                 validation_data=(x_test, y_test),
                                                 callbacks=[lr_scheduler],
                                                 verbose=verbose)
        self.accuracy = train_history.history['val_acc'][-1]
        self.max_accuracy = max(train_history.history['val_acc'])
        self.model_size = self.model.count_params() / 1000000

    def search_blocks(self, structures):
        '''
        A utility function to perform iNAS, this function get a list of structures, trains all models for a given number
            of epochs and return the best structure in terms of validation accuracy.
        :param structures: a list of lists, each nested list represent a structure.
        :return: the structure that best performed over the val set ( alist)
        '''
        results = []
        for struct in structures:
            print("arcitecture search for {}".format(struct))
            if str(struct) in self.search_res.keys():
                results.append(self.search_res[str(struct)])
            else:
                self.structure = struct
                self.model = self._build_model(blocks=self.structure)
                self.train(50, validation=False)
                results.append(self.max_accuracy)
                self.search_res[str(struct)] = self.max_accuracy

            print("Accuracy for {} is {}".format(struct, results[-1]))

        best_structure = np.argmax(np.array(results))
        return structures[best_structure]
