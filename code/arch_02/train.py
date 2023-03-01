"""
    Author: Daniel Al Mouiee
    Date:   27/03/2021

    Script to train a CNN to classify histologically stained images of feline retinae into 4 different classes.

    Architrecture 02, refer to paper for further details
"""

import os, sys
import dataset
import tensorflow as tf

from numpy.random import seed

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# train path: /srv/scratch/z5114185/thesis/datasetV1.1

# Randomisation
seed(1)
tf.set_random_seed(2)

# Hyperparameters
validation_size = 0.2
img_size = 128
num_channels = 3
batch_size = 256
learningRate = 1e-3

# Architecture parameters
filter_size_conv1 = 3
num_filters_conv1 = 64
filter_size_conv2 = 3
num_filters_conv2 = 64
filter_size_conv3 = 3
num_filters_conv3 = 92
filter_size_conv4 = 3
num_filters_conv4 = 92
filter_size_conv5 = 3
num_filters_conv5 = 128
filter_size_conv6 = 3
num_filters_conv6 = 128

fc_layer_size = 128


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(
    input, num_input_channels, conv_filter_size, num_filters
):

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]
    )
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(
        input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME"
    )

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(
        value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def show_progress(
    session,
    epoch,
    feed_dict_train,
    feed_dict_validate,
    accuracy,
    merged,
    val_accuracy,
    val_loss,
    train_writer,
    test_writer,
    i,
):
    msg = "Training Epoch {0}, Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    summary_train, train_acc = session.run(
        [merged, accuracy], feed_dict=feed_dict_train
    )
    train_writer.add_summary(summary_train, i)

    summary_test, val_acc = session.run(
        [merged, val_accuracy], feed_dict=feed_dict_validate
    )
    test_writer.add_summary(summary_test, i)
    print(msg.format(epoch + 1, train_acc, val_acc, val_loss))

    # val_acc = session.run(accuracy, feed_dict=feed_dict_validate)


def train(
    session,
    data,
    x,
    y_true,
    cost,
    optimiser,
    accuracy,
    merged,
    val_accuracy,
    train_writer,
    test_writer,
    saver,
    num_iteration,
):
    total_iterations = 0

    for i in range(total_iterations, total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(
            batch_size
        )

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
        session.run(optimiser, feed_dict=feed_dict_tr)
        if i % 10:
            saver.save(session, "./arch_02")
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(
                session,
                epoch,
                feed_dict_tr,
                feed_dict_val,
                accuracy,
                merged,
                val_accuracy,
                val_loss,
                train_writer,
                test_writer,
                i,
            )

    total_iterations += num_iteration


## start of prog
def main():

    # Prepare input data
    train_path = sys.argv[1]
    classes = os.listdir(train_path)
    num_classes = len(classes)

    # We shall load all the training and validation images and labels into memory using openCV
    # and use that during training
    data = dataset.read_train_sets(
        train_path, img_size, classes, validation_size=validation_size
    )

    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    # TF variables
    x = tf.placeholder(
        tf.float32, shape=[None, img_size, img_size, num_channels], name="x"
    )
    # labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
    y_true_cls = tf.argmax(y_true, dimension=1)

    # CNN Archiecture
    layer_conv1 = create_convolutional_layer(
        input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1,
    )

    layer_conv2 = create_convolutional_layer(
        input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2,
    )

    layer_conv3 = create_convolutional_layer(
        input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3,
    )

    layer_conv4 = create_convolutional_layer(
        input=layer_conv3,
        num_input_channels=num_filters_conv3,
        conv_filter_size=filter_size_conv4,
        num_filters=num_filters_conv4,
    )

    layer_conv5 = create_convolutional_layer(
        input=layer_conv4,
        num_input_channels=num_filters_conv4,
        conv_filter_size=filter_size_conv5,
        num_filters=num_filters_conv5,
    )

    layer_conv6 = create_convolutional_layer(
        input=layer_conv5,
        num_input_channels=num_filters_conv5,
        conv_filter_size=filter_size_conv6,
        num_filters=num_filters_conv6,
    )

    layer_flat = create_flatten_layer(layer_conv6)

    layer_fc1 = create_fc_layer(
        input=layer_flat,
        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
        num_outputs=fc_layer_size,
        use_relu=True,
    )

    layer_fc2 = create_fc_layer(
        input=layer_fc1,
        num_inputs=fc_layer_size,
        num_outputs=num_classes,
        use_relu=False,
    )

    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # Hyperparameters
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=layer_fc2, labels=y_true
    )
    cost = tf.reduce_mean(cross_entropy)
    optimiser = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TF Summary Variables
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("val_accuracy", val_accuracy)
    tf.summary.scalar("cost", cost)
    merged = tf.summary.merge_all()

    # Tensorboard summary file writers
    train_writer = tf.summary.FileWriter("summary/train_arch_02", session.graph)
    test_writer = tf.summary.FileWriter("summary/test_arch_02")

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # Train model
    train(
        session,
        data,
        x,
        y_true,
        cost,
        optimiser,
        accuracy,
        merged,
        val_accuracy,
        train_writer,
        test_writer,
        saver,
        num_iteration=2000,
    )

    # Save a model checkpoint
    saver.save(session, "./arch_02")


main()
