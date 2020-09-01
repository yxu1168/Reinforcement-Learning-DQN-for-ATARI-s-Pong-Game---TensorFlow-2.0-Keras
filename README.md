## Reinforcement-Learning-DQN-for-ATARI-s-Pong-Game---TensorFlow-2.0-Keras
### Implemented CNN DQN, Double DQN and Dueling Double DQN Models to Atari's Pong Game

1). Data Preprocessing: 
    Pong game original input shape is (210, 160, 3); Some irrelevant pixels were cropped out from the original image; The color of the image is not important for the decision of actions, thus the screen was further decolored to single channel. The image was finally resized to (84, 84, 1). 

2). CNN DQN Model:
    Three Convolutional Neural Network layers are added to the DQN model. Train the model with preprocessed input data.
    
3). Double DQN and Dueling Double DQN models are built on CNN DQN network.

4). This repository includes three Colab Notebooks: CNN DQN, Double DQN and Dueling Double DQN. (Removed basic three dense layers DQN)

### DQN Model layers:
First CNN layer: neurons 16, kernel size = 8;
Second CNN layer: neurons 32, kernel size = 4;
Third CNN layer: neurons 32, kernel size = 3;
Neurons of first two dense layer: 64, output layer: 6;
Activation: relu.

The total parameters for Basic DQN is 6,455,814; reduced 14 times by DQN + data preprocessing: 456,198; and further reduced by CNN DQN: 123,478.

### Colab Notebook
The Notebook code is modified from Aurélien Geron’s GitHub: https://github.com/ageron/handson-ml2 :
18_reinforcement_learning.ipynb; The original Notebook is for a simple CartPole game:  env = gym.make("CartPole-v1")

Some necessary changes:

env Change to be:  env = gym.make('PongNoFrameskip-v4’)

Original input and output:
input_shape = [4];
n_outputs = 2 

Change to be:
input_shape = [210, 160, 3]; 
n_outputs = 6 ( 6 actions: 'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE’)

And other corresponding adjustments.

