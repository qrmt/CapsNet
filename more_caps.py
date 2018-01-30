
# coding: utf-8

# In[1]:


from cifar10 import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import datetime
import sys
import csv


import tensorflow as tf



dest_dir = sys.argv[1]

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

model_name = dest_dir+'/model'

# In[2]:


maybe_download_and_extract()
class_names = load_class_names()

train_data = load_training_data()
train_images_original = train_data[0]
train_labels_original = train_data[1]


# In[3]:


test_data = load_test_data()
test_images = test_data[0]
test_labels = test_data[1]


# In[4]:


# Data centering and data normalization.
train_images_original = train_images_original.astype('float32')
mean = np.mean(train_images_original)  # mean for data centering
std = np.std(train_images_original)  # std for data normalization

train_images_original -= mean
train_images_original /= std


# In[5]:


# Split to train and validation data
split = 0.9
split_loc = int(split * len(train_images_original))
              
validation_images = train_images_original[split_loc:]
validation_labels = train_labels_original[split_loc:]

train_images = train_images_original[:split_loc]
train_labels = train_labels_original[:split_loc]


# In[6]:


# Horizontal flip to all training data
train_images_flip = np.flip(train_images, axis=2)
train_images = np.concatenate((train_images, train_images_flip), axis=0)
train_labels = np.concatenate((train_labels, train_labels), axis=0)


# ## Tensorflow implementation

# In[7]:


X = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X')


# In[8]:


# 32 maps of 6x6 capsules, outputs 8D activation vector
caps1_maps_amount = 64
caps1_caps_amount = caps1_maps_amount * 8 * 8
caps1_dims = 8

# First apply two regular convolutional layers
C1 = tf.layers.conv2d(X, 
                      filters=256, 
                      kernel_size = 9, 
                      strides=1, 
                      padding='valid', 
                      activation=tf.nn.relu, 
                      name='C1')

C2 = tf.layers.conv2d(C1, 
                      filters=(caps1_maps_amount*caps1_dims), 
                      kernel_size = 9, 
                      strides=2, 
                      padding='valid', 
                      activation=tf.nn.relu, 
                      name='C2')


# In[9]:


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


# In[10]:


# Reshape to capsule
caps1_input = tf.reshape(C2, [-1, caps1_caps_amount, caps1_dims],
                       name="caps1_input")

# Calculate capsule output
caps1_output = squash(caps1_input, name='caps1_output')


# In[11]:


# Image capsule layer
caps2_caps_amount = 10
caps2_dims = 16

init_sigma = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_caps_amount, caps2_caps_amount, caps2_dims, caps1_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


# In[12]:


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_caps_amount, 1, 1],
                             name="caps1_output_tiled")


# In[13]:


caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")


# In[14]:


caps2_predicted


# ## Routing by agreement

# ### Dynamic loop for routing

# In[15]:


# Initialize routing weights to zero
raw_weights = tf.zeros([batch_size, caps1_caps_amount, caps2_caps_amount, 1, 1],
                       dtype=np.float32, name="raw_weights")


# In[16]:


def routingNode(capsule_output, W, caps_prediction, raw_weights_initial=None):
    # X: Tiled output of a capsule layer
    # W: Current routing weights
    # caps_prediction: current prediction for the capsule layer outputs
    # raw_weights_initial: If the raw weights are already calculated, use them (First iteration)
    
    if raw_weights_initial is not None:
        raw_weights = raw_weights_initial
    
    else:
        agreement = tf.matmul(caps_prediction, capsule_output,
                          transpose_a=True, name="agreement")


        raw_weights = tf.add(W, agreement)

    routing_weights = tf.nn.softmax(raw_weights, dim=2)

    weighted_predictions = tf.multiply(routing_weights, caps_prediction)

    weighted_sum = tf.reduce_sum(weighted_predictions,
                                         axis=1, keep_dims=True)

    caps_output = squash(weighted_sum, axis=-2)
    
    return caps_output, raw_weights


# In[17]:


# First routing node
caps2_output, raw_weights = routingNode(None, None, caps2_predicted, raw_weights)

# Loop for consequent routing nodes
total_routes = 4
for i in range(1,total_routes):
    caps2_output_tiled = tf.tile(caps2_output, [1, caps1_caps_amount, 1, 1, 1])

    caps2_output, raw_weights = routingNode(caps2_output_tiled, raw_weights, caps2_predicted)


# ## Estimating class probs

# In[18]:


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


# In[19]:


y_prob = safe_norm(caps2_output, axis=-2, name="y_prob")
y_prob_argmax = tf.argmax(y_prob, axis=2, name="y_prob")
y_pred = tf.squeeze(y_prob_argmax, axis=[1,2], name="y_pred")


# ## Labels

# In[20]:


y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")


# In[21]:


m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5


# In[22]:


T = tf.one_hot(y, depth=caps2_caps_amount, name="T") # One hot encode labels


# ## Reconstruction as regularization

# In[23]:


# Compute output vector for each output capsule and instance
print(caps2_output)

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")


# In[24]:


present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")


# In[25]:


absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")


# In[26]:


L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")


# In[27]:


margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


# ## Mask
# Only send outputs to the reconstruction network from the capsule to which the image belongs

# In[28]:


mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")


# In[29]:


reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")


# In[30]:


reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_caps_amount,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_caps_amount, 1, 1],
    name="reconstruction_mask_reshaped")


# In[31]:


reconstruction_mask


# In[32]:


caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")


# In[33]:


decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_caps_amount * caps2_dims],
                           name="decoder_input")


# ## Decoder
# Two fully connected ReLU followed with dense output sigmoid layer

# In[35]:


n_hidden1 = 512
n_hidden2 = 1024
n_output = 32 * 32 * 3


# In[36]:


with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")


# ## Reconstruction loss

# In[37]:


X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")


# ## Final loss

# In[38]:


alpha = 0.0001

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")


# ## Calculate accuracy

# In[39]:


correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


# ## Training ops

# In[40]:


optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")


# In[41]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# # Training

# In[42]:


def reset_batch(images, labels):
    # Shuffles the data
    p = np.random.permutation(len(images))
    shuffled_images = images[p]
    shuffled_labels = labels[p]
    return shuffled_images, shuffled_labels

def get_next_batch(batch_size, iteration, epoch_images, epoch_labels):
    batch_location = (iteration - 1)*batch_size
    batch_images = epoch_images[batch_location:batch_location+batch_size]
    batch_labels = epoch_labels[batch_location:batch_location+batch_size]
    
    return batch_images, batch_labels


# In[43]:


n_epochs = 10
batch_size = 100
restore_checkpoint = True

n_iterations_per_epoch = len(train_images) // batch_size
n_iterations_validation = len(validation_images) // batch_size
best_loss_val = np.infty
checkpoint_path = model_name

training_loss_values = []


with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    
    for epoch in range(n_epochs):
        epoch_start_time = timeit.default_timer()
        epoch_images_train, epoch_labels_train = reset_batch(train_images, train_labels)
        epoch_images_validation, epoch_labels_validation = reset_batch(validation_images, validation_labels)

        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = get_next_batch(batch_size, iteration, 
                                              epoch_images_train, epoch_labels_train)
            
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 32, 32, 3]),
                           y: y_batch,
                           mask_with_labels: True})
            training_loss_values.append(loss_train)
            
            iteration_time = timeit.default_timer() - epoch_start_time
            print_line = "\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f} Seconds: {:.1f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train,
                      iteration_time)
            print(print_line, end="")
            sys.__stdout__.write(print_line)

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = get_next_batch(batch_size, iteration, 
                                              epoch_images_validation, epoch_labels_validation)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 32, 32, 3]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print_line = "\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation)
            print(print_line, end=" " * 10)
            sys.__stdout__.write(print_line)
            
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        epoch_time = (timeit.default_timer() - epoch_start_time)/60.0
        print_line = "Epoch: {} Train loss: {:.6f} Val accuracy: {:.4f}%  Loss: {:.6f} Minutes: {:.2f}{}".format(
            epoch + 1, loss_train, acc_val * 100, loss_val, epoch_time,
            " (improved)" if loss_val < best_loss_val else "")
        print(print_line)
        sys.__stdout__.write(print_line)
        
        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val

