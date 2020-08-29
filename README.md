## Reinforcement-Learning-DQN-for-ATARI-s-Pong-Game---TensorFlow-2.0-Keras
### Build DQN Model: Three approaches
1). Basic DQN with three dense layers: 
    First two dense layers with 64 neurons and output layer with 6. Input is original shape (210, 160, 3). 

2). Basic DQN with Data Preprocessing: 
    Some irrelevant pixels were cropped out from the original image. The color of the image is not important for the decision of         actions, thus the screen was further decolored to single channel. The image was finally resized to (84, 84, 1). Then run the         same model as approach 1.

3). CNN DQN Model:
    Three Convolutional Neural Network layers are added to the DQN model. Train the model with preprocessed input data. 

### DQN Model layers:
Neurons of first two dense layer: 64, output layer: 6;
First CNN layer: neurons 16, kernel size = 8;
Second CNN layer: neurons 32, kernel size = 4;
Third CNN layer: neurons 32, kernel size = 3;
Activation: relu.

The total parameters for Basic DQN is 6,455,814; reduced 14 times by DQN + data preprocessing: 456,198; and further reduced by CNN DQN: 123,478.

### Colab Notebook
The Notebook code is modified from Aurélien Geron’s GitHub: https://github.com/ageron/handson-ml2 :
18_reinforcement_learning.ipynb: Deep Q-Network part. Original env:  env = gym.make("CartPole-v1")

Major changes:

env Change to be:  env = gym.make('PongNoFrameskip-v4’)

Original input and output:
input_shape = [4];
n_outputs = 2 

Change to be:
input_shape = [210, 160, 3]; 
n_outputs = 6 ( 6 actions: 'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE’)

To make input into neural network, add a Flatten layer before the dense layers:
keras.layers.Flatten(input_shape=input_shape)

And other corresponding adjustments.

