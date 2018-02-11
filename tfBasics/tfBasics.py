import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

np.random.seed(42)
tf.set_random_seed(42)

def basic():
	hello = tf.constant("Hello ")
	world = tf.constant("world!")

	print(type(hello))
	print(hello)

	with tf.Session() as sess:
		res = sess.run(hello + world)
		print(res)

		a1 = tf.constant(10)
		b1 = tf.constant(20)

		res = sess.run(a1 + b1)
		print(res)

		const = tf.constant(10)
		mat = tf.fill((4,4),10)
		zeros = tf.zeros((4,4))
		ones = tf.ones((4,4))
		myrandn = tf.random_normal((4,4), mean=0,stddev=1.0)
		myrandu = tf.random_uniform((4,4), minval=0,maxval=1)

		my_ops = [const, mat, zeros, ones, myrandn, myrandu]

		for op in my_ops:
			print("{0}:\n{1}".format(str(op), sess.run(op)))

		a = tf.constant(np.arange(1,5).reshape(2,2).astype(np.float32))
		print(sess.run(a))
		print(a.get_shape())
		b = tf.constant((np.array([10, 100]).reshape(2,1).astype(np.float32)))
		print(sess.run(b))
		print(b.get_shape())
		result = tf.matmul(a, b)
		print(sess.run(result))

def graph():
	sess = tf.Session()
	n1 = tf.constant(1)
	n2 = tf.constant(2)
	n3 = n1 + n2
	result = sess.run(n3)
	print(result)
	print(tf.get_default_graph())
	g = tf.Graph()
	print(g)
	graph_one = tf.get_default_graph()
	print(graph_one)
	graph_two = tf.Graph()

	#true
	with graph_two.as_default():
		print(graph_two is tf.get_default_graph())

	#false
	graph_two is tf.get_default_graph()

	sess.close()

def VariablesPlaceholders():
	
	sess = tf.Session()
	# reserve memory for mtrix
	my_tensor = tf.random_uniform((4,4),0,1)
	print(my_tensor)
	# reserve memory for variable that inits as matrix
	my_var = tf.Variable(initial_value=my_tensor)
	print(my_var)


	# BEFORE use any variables or placeholder INIT variables with tf.global_variables_initializer()
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(my_var))


	sess.close()

def simpleNeuron():

	# 1) Build Graph
	# 2) Initiate the Session
	# 3) Feed Data and get Output

	sess = tf.Session()

	rand_a = np.random.uniform(0,100,(5,5))
	rand_b = np.random.uniform(0,100,(5,1))

	# Create placeholders for this objects

	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)

	# Create operations add and mult

	add_op = a + b
	mul_op = a * b

	add_result = sess.run(add_op, feed_dict={a:10,b:20})
	print(add_result)
	print("\n{0}\n=^_^=\n{1}\n".format(rand_a, rand_b))
	add_res = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
	print(add_res)
	mul_res = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
	print("\n",mul_res)

	sess.close()

def simpleNN():
	
	sess = tf.Session()

	n_features = 10
	n_dense_neurons = 3

	x = tf.placeholder(tf.float32, (None, n_features))
	W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

	b = tf.Variable(tf.ones([n_dense_neurons]))

	xW = tf.matmul(x, W)

	z = tf.add(xW, b)

	a = tf.sigmoid(z)

	init = tf.global_variables_initializer()

	sess.run(init)

	y_out = sess.run(a, feed_dict={x:np.random.random([1, n_features])})
	print(y_out)

	sess.close()

def simpleRegression():

	sess = tf.Session()

	x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)
	print(x_data)
	y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

	# plt.plot(x_data, y_label, "+")
	# plt.show()

	# y = m*x + b
	m = tf.Variable(0.44)
	b = tf.Variable(0.87)

	err = 0

	# Cost function
	for x, y in zip(x_data, y_label):

		y_pred = m*x + b

		err += (y - y_pred)**2

	# Optimize

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

	train = optimizer.minimize(err)

	init = tf.global_variables_initializer()

	sess.run(init)

	traininig_step = 400
	for i in range(traininig_step):

		sess.run(train)

	final_slope, finale_intercept = sess.run([m,b])

	x_test = np.linspace(-1,11,10)
	y_pred_plot = final_slope * x_test + finale_intercept

	plt.plot(x_test, y_pred_plot,'r')
	plt.plot(x_data, y_label, '*')
	plt.show()



	sess.close()

def main():
	# graph()
	# VariablesPlaceholders()
	# simpleNeuron()
	# simpleNN()
	simpleRegression()


if __name__ == '__main__':
	main()