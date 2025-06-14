// App.jsx
import React, { useEffect, useState } from "react";
import NeuralNetwork from "./Lib/nn"; // Your library file

const App = () => {
  const [results, setResults] = useState([]);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    const runXOR = async () => {
      // XOR data
      const inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ];
      const targets = [[0], [1], [1], [0]];

      // Init NN
      const nn = new NeuralNetwork({
        inputSize: 2,
        hiddenSize: 4,
        outputSize: 1,
        learningRate: 0.1,
        activationHidden: "tanh",
        activationOutput: "sigmoid",
      });

      // Train for 1000 epochs
      for (let i = 0; i < 1000; i++) {
        nn.trainBatch(inputs, targets);
        if (i % 100 === 0) {
          console.log(`Training... epoch ${i}`);
          setEpoch(i);
        }
      }

      // Predict
      const output = inputs.map((inp) => {
        try {
          const prediction = nn.predict(inp); // returns Float32Array
          const [pred] = Array.from(prediction || []); // ensures fallback to empty array
          return {
            input: inp,
            output: pred !== undefined ? parseFloat(pred.toFixed(3)) : "N/A",
          };
        } catch (err) {
          console.error("Prediction error:", err);
          return {
            input: inp,
            output: "error",
          };
        }
      });
      

      setResults(output);
    };

    runXOR();
  }, []);

  return (
    <div style={{ fontFamily: "monospace", padding: "2rem" }}>
      <h1>🧠 XOR Neural Network Demo</h1>
      <p>Trained for {epoch}+ epochs</p>
      <table border="1" cellPadding="10" style={{ marginTop: "1rem" }}>
        <thead>
          <tr>
            <th>Input</th>
            <th>Output (Prediction)</th>
          </tr>
        </thead>
        <tbody>
          {results.map((res, i) => (
            <tr key={i}>
              <td>[{res.input.join(", ")}]</td>
              <td>{res.output}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default App;
