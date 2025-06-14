import * as tf from "@tensorflow/tfjs";

class NeuralNetwork {
  constructor(numInputs, numHidden, numOutputs) {
    this.input_Nodes = numInputs;
    this.hidden_Nodes = numHidden;
    this.output_Nodes = numOutputs;

    this.weights_ih = tf.variable(
      tf.randomNormal([this.input_Nodes, this.hidden_Nodes])
    );
    this.weights_ho = tf.variable(
      tf.randomNormal([this.hidden_Nodes, this.output_Nodes])
    );
    this.bias_h = tf.variable(tf.randomNormal([1, this.hidden_Nodes]));
    this.bias_o = tf.variable(tf.randomNormal([1, this.output_Nodes]));

    this.learning_rate = 0.1;
  }

  async feedForward(input_array) {
    const inputs = tf.tensor2d([input_array]);
    const hidden = inputs.matMul(this.weights_ih).add(this.bias_h).sigmoid();
    const outputs = hidden.matMul(this.weights_ho).add(this.bias_o).sigmoid();

    const result = await outputs.data(); // async version!
    inputs.dispose();
    hidden.dispose();
    outputs.dispose();
    return result;
  }

  train(input_array, target_array) {
    tf.tidy(() => {
      const inputs = tf.tensor2d([input_array]);
      const targets = tf.tensor2d([target_array]);

      const hidden = inputs.matMul(this.weights_ih).add(this.bias_h).sigmoid();
      const outputs = hidden.matMul(this.weights_ho).add(this.bias_o).sigmoid();

      const output_errors = targets.sub(outputs);

      const gradients = outputs
        .mul(tf.scalar(1).sub(outputs))
        .mul(output_errors)
        .mul(this.learning_rate);

      const weights_ho_deltas = hidden.transpose().matMul(gradients);
      this.weights_ho.assign(this.weights_ho.add(weights_ho_deltas));
      this.bias_o.assign(this.bias_o.add(gradients.mean(0)));

      const hidden_errors = output_errors.matMul(this.weights_ho.transpose());
      const hidden_gradient = hidden
        .mul(tf.scalar(1).sub(hidden))
        .mul(hidden_errors)
        .mul(this.learning_rate);

      const weights_ih_deltas = inputs.transpose().matMul(hidden_gradient);
      this.weights_ih.assign(this.weights_ih.add(weights_ih_deltas));
      this.bias_h.assign(this.bias_h.add(hidden_gradient.mean(0)));
    });
  }

  trainBatch(input_batch, target_batch) {
    tf.tidy(() => {
      const inputs = tf.tensor2d(input_batch); // shape [batch, 2]
      const targets = tf.tensor2d(target_batch); // shape [batch, 1]

      const hidden = inputs.matMul(this.weights_ih).add(this.bias_h).sigmoid();
      const outputs = hidden.matMul(this.weights_ho).add(this.bias_o).sigmoid();

      const output_errors = targets.sub(outputs);

      const gradients = outputs
        .mul(tf.scalar(1).sub(outputs))
        .mul(output_errors)
        .mul(this.learning_rate);

      const weights_ho_deltas = hidden.transpose().matMul(gradients);
      this.weights_ho.assign(this.weights_ho.add(weights_ho_deltas));
      this.bias_o.assign(this.bias_o.add(gradients.mean(0)));

      const hidden_errors = output_errors.matMul(this.weights_ho.transpose());
      const hidden_gradient = hidden
        .mul(tf.scalar(1).sub(hidden))
        .mul(hidden_errors)
        .mul(this.learning_rate);

      const weights_ih_deltas = inputs.transpose().matMul(hidden_gradient);
      this.weights_ih.assign(this.weights_ih.add(weights_ih_deltas));
      this.bias_h.assign(this.bias_h.add(hidden_gradient.mean(0)));
    });
  }
}

export default NeuralNetwork;
