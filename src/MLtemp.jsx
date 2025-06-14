import React, { useEffect, useState } from "react";
import NeuralNetwork from "./Lib/nn"; // Import your custom NN library

// Generalized App Component for ML Projects
const MLAppTemplate = ({
  neuralNetworkConfig,    // Configuration for NN (inputSize, hiddenSize, outputSize, etc.)
  trainingData,           // Input data for training
  targetData,             // Target data for training (labels)
  totalEpochs = 5000,     // Number of epochs to train
  batchSize = 100,        // Number of epochs per batch
  progressUpdateInterval = 10, // Interval between each progress update
}) => {
  const [loading, setLoading] = useState(true);
  const [epoch, setEpoch] = useState(0);
  const [progress, setProgress] = useState(0);
  const [outputs, setOutputs] = useState([]);

  const { inputSize, hiddenSize, outputSize, learningRate, activationHidden, activationOutput } = neuralNetworkConfig;

  const runTraining = () => {
    const nn = new NeuralNetwork({
      inputSize,
      hiddenSize,
      outputSize,
      learningRate,
      activationHidden,
      activationOutput,
    });

    let currentEpoch = 0;

    // Training loop with a delay (chunked)
    const trainingInterval = setInterval(() => {
      for (let i = 0; i < batchSize; i++) {
        nn.trainBatch(trainingData, targetData); // Train the batch synchronously
        currentEpoch++;

        // Update progress
        const progressPercentage = Math.floor((currentEpoch / totalEpochs) * 100);
        setProgress(progressPercentage);

        if (currentEpoch % 100 === 0) console.log(`Training... epoch ${currentEpoch}`);

        setEpoch(currentEpoch); // Update state after batch
        if (currentEpoch >= totalEpochs) {
          clearInterval(trainingInterval); // Stop training when done
          // Once training is done, make predictions
          const output = trainingData.map((inp) => {
            const prediction = nn.predict(inp);
            const [pred] = prediction || [];
            return {
              input: inp,
              output: pred !== undefined ? parseFloat(pred.toFixed(3)) : "N/A",
            };
          });

          setOutputs(output); // Set predictions after training is done
          setLoading(false); // Hide loading state
        }
      }
    }, progressUpdateInterval); // Run the training every 10ms (adjust as needed)
  };

  useEffect(() => {
    runTraining(); // Start the training process when component mounts
  }, []);

  return (
    <div style={{ fontFamily: "monospace", padding: "2rem" }}>
      <h1>🧠 Machine Learning Model Demo</h1>
      {loading ? (
        <div style={{ fontSize: "1.2rem", color: "#555" }}>
          <span className="loader" style={{ marginRight: "1rem" }}>
            🔄
          </span>
          Training the model, please wait...
          <p>Progress: {progress}%</p>
        </div>
      ) : (
        <div>
          <p>Trained for {epoch}+ epochs</p>
          <table border="1" cellPadding="10" style={{ marginTop: "1rem" }}>
            <thead>
              <tr>
                <th>Input</th>
                <th>Output (Prediction)</th>
              </tr>
            </thead>
            <tbody>
              {outputs.map((res, i) => (
                <tr key={i}>
                  <td>[{res.input.join(", ")}]</td>
                  <td>{res.output}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default MLAppTemplate;
