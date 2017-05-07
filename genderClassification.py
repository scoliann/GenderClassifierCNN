import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score

# Read in photos for each class and encode
maleFolder = 'testData\male'
femaleFolder = 'testData\\female'

maleClassLabels = ['male' for f in listdir(maleFolder) if isfile(join(maleFolder, f))]
femaleClassLabels = ['female' for f in listdir(femaleFolder) if isfile(join(femaleFolder, f))]

malePhotos = [join(maleFolder, f) for f in listdir(maleFolder) if isfile(join(maleFolder, f))]
femalePhotos = [join(femaleFolder, f) for f in listdir(femaleFolder) if isfile(join(femaleFolder, f))]
encodedMalePhotos = [tf.gfile.FastGFile(photo, 'rb').read() for photo in malePhotos]
encodedFemalePhotos = [tf.gfile.FastGFile(photo, 'rb').read() for photo in femalePhotos]

X = encodedMalePhotos + encodedFemalePhotos
y = maleClassLabels + femaleClassLabels

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

# Make predictions
predictionList = []
with tf.Session() as sess:

	# Feed the image_data as input to the graph and get first prediction
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	# Iterate over all images and make predictions
	imageCounter = 0
	for image_data in X:

		# Print image coutner to terminal
		imageCounter += 1
		print('On Image ' + str(imageCounter) + '/' + str(len(X)))

		# Make a prediction
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		# Get the predicted class and add it to list of predictions
		prediction = label_lines[top_k[0]]
		predictionList.append(prediction)

# Calculate classification accuracy
accuracy = 100 * accuracy_score(y, predictionList)
print('Accuracy: \t' + str(accuracy) + '%')





