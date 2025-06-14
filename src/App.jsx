import React from "react";
import MLAppTemplate from "./MLtemp"; // Import your reusable template
import NeuralNetwork from "./Lib/nn"; // Your custom NN library

const App = () => {
  const neuralNetworkConfig = {
    inputSize: 2,
    hiddenSize: 6,
    outputSize: 1,
    learningRate: 0.1,
    activationHidden: "tanh",
    activationOutput: "sigmoid",
  };

  const trainingData = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];

  const targetData = [[0], [1], [1], [0]]; // XOR targets

  return (
    <MLAppTemplate
      neuralNetworkConfig={neuralNetworkConfig}
      trainingData={trainingData}
      targetData={targetData}
    />
  );
};

export default App;
