import { useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import NeuralNetwork from "./Lib/nn";

const training_data = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

function getRandomTrainingData(data) {
  const index = Math.floor(Math.random() * data.length);
  return data[index];
}

function App() {
  const hasRun = useRef(false);

  useEffect(() => {
    const run = async () => {
      if (hasRun.current) return;
      hasRun.current = true;

      await tf.ready();

      const nn = new NeuralNetwork(2, 2, 1);

      // Training faster: full batch per iteration
      for (let i = 0; i < 100000; i++) {
        const inputs = training_data.map((d) => d.inputs); // shape: [4, 2]
        const targets = training_data.map((d) => d.targets); // shape: [4, 1]
        nn.trainBatch(inputs, targets);
      }

      console.log("🧠 Training complete! Here's the XOR logic:");

      const testInputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ];

      for (const input of testInputs) {
        const output = await nn.feedForward(input);
        console.log(`Input: ${input} → Output: ${output[0].toFixed(3)}`);
      }
    };

    run();
  }, []);

  return <h1>XOR Neural Network Trained 💡 Open console</h1>;
}

export default App;
