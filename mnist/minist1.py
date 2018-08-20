import loader
import numpy as np
import tensorflow as tf

X_train, y_train, X_valid, y_valid, X_test = loader.load_data()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 1.0

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


def neural_layer(X, n_neurons, activation=None):
    # shape n_inputs * n_outpu
  n_inputs = int(X.get_shape()[1])
  stddev = 2 / np.sqrt(n_inputs)
  init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
  W = tf.Variable(init, name="kernel")
  b = tf.Variable(tf.zeros(n_neurons), name="bias")
  Z = tf.add(tf.matmul(X, W), b)
  if activation is not None:
    return activation(Z)
  else:
    return Z


with tf.name_scope("dnn"):
  hidden1 = neural_layer(X, n_hidden1, tf.nn.relu)
  hidden2 = neural_layer(hidden1, n_hidden2, tf.nn.relu)
  logits = neural_layer(hidden2, n_outputs)

with tf.name_scope("loss"):
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y, name="xentopy")
  loss = tf.reduce_mean(xentropy)

with tf.name_scope("train"):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
  correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 2
batch_size = 100


def shuffle_data(X, y, batch_size):
  rnd_idx = np.random.permutation(len(X))
  n_batchs = len(X) // batch_size
  for batch_idx in np.array_split(rnd_idx, n_batchs):
    yield X[batch_idx], y[batch_idx]


with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):
    for X_batch, y_batch in shuffle_data(X_train, y_train, batch_size):
      # How to get loss number?
      _, loss_val = sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch})
    acc_batch = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
    acc_valid = sess.run(accuracy, feed_dict={X: X_valid, y: y_valid})
    print(epoch, "batch_accuracy: ", acc_batch, " valid_accuracy: ", acc_valid)
  save_path = saver.save(sess, "./parameters/my_model1.ckpt")


with tf.Session() as sess:
  saver.restore(sess, "./parameters/my_model1.ckpt")
  X_new_scaled = X_test[:20]
  Z = sess.run(logits, feed_dict={X: X_new_scaled})
  y_pred = np.argmax(Z, axis=1)

print(y_pred)

file_writer = tf.summary.FileWriter("./logs/nn1", tf.get_default_graph())
file_writer.close()






