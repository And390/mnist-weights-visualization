# Neural Network Weights Visualization Demo
The purpose of this demo is to show what patterns neural network learn during a MNIST number recognition task
(a well-known set of images of 28x28 pixels size with handwritten digits from 0 to 9).
It uses the simplest neural network with one inner layer (of configurable size) and an output layer of 10 neurons (one for each digit).
For each neuron in the inner layer it displays a 28x28 image where black and white (or red and green if you choose) mean negative and positive weights.
So for each neuron you see what pixel patterns it recognizes.

Uses TensorFlow.js.

[Live demo](http://mnist-patterns.my.to/)