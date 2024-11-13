# NeuralNet

NeuralNet is a C library for training Feed-Forward Neural Networks. Currently still under development, but it can train to >95% accuracy using the ReLU activation function.

## Usage

Initialize a struct for all the training data, and a struct for the network.

```C
int numLayers = 4;
int numNodes[] = {2, 3, 3, 1};
fnnData * trainingData = load_fnnData(numInputs, numOutputs, "filename.csv");
fnnNet * neuralNet = initNet(numLayers, numNodes);
```

Then train the network using trainNet, and print out the resulting weights.

```C
int numEpochs = 3000;
trainNet(trainingData, neuralNet, numEpochs);
printWeights(neuralNet);
```
