import React, { useEffect, useState } from "react";
import NeuralNetwork from "./Lib/nn"; // Your custom NN library

const App = () => {
  const [loading, setLoading] = useState(true);
  const [epoch, setEpoch] = useState(0);
  const [outputs, setOutputs] = useState([]);

  const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];

  const targets = [[0], [1], [1], [0]]; // XOR targets

  const runXOR = async () => {
    const nn = new NeuralNetwork({
      inputSize: 2,
      hiddenSize: 6,
      outputSize: 1,
      learningRate: 0.1,
      activationHidden: "tanh",
      activationOutput: "sigmoid",
    });

    // Train the network
    for (let i = 0; i < 5000; i++) {
      nn.trainBatch(inputs, targets);
      if (i % 100 === 0) console.log(`Training... epoch ${i}`);
      setEpoch(i);
    }

    // Predict
    const output = inputs.map((inp) => {
      const prediction = nn.predict(inp); // returns Float32Array
      const [pred] = prediction || [];
      return {
        input: inp,
        output: pred !== undefined ? parseFloat(pred.toFixed(3)) : "N/A",
      };
    });

    setOutputs(output);
    setLoading(false); // hide loader after training
  };

  useEffect(() => {
    runXOR();
  }, []);

  return (
    <div style={{ fontFamily: "monospace", padding: "2rem" }}>
      <h1>🧠 XOR Neural Network Demo</h1>
      {loading ? (
        <div style={{ fontSize: "1.2rem", color: "#555" }}>
          <span className="loader" style={{ marginRight: "1rem" }}>
            🔄
          </span>
          Training the model, please wait...
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

export default App;
