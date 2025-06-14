import * as tf from "@tensorflow/tfjs";

class NeuralNetwork {
  constructor(config) {
    const {
      inputSize,
      hiddenSize = 4,
      outputSize,
      learningRate = 0.1,
      activationHidden = "tanh",
      activationOutput = "sigmoid",
    } = config;

    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.activationHidden = activationHidden;
    this.activationOutput = activationOutput;

    // Initialize weights and biases
    this.weights_ih = tf.variable(
      tf.randomNormal([this.inputSize, this.hiddenSize])
    );
    this.weights_ho = tf.variable(
      tf.randomNormal([this.hiddenSize, this.outputSize])
    );
    this.bias_h = tf.variable(tf.zeros([1, this.hiddenSize]));
    this.bias_o = tf.variable(tf.zeros([1, this.outputSize]));
  }

  _activate(tensor, fn) {
    switch (fn) {
      case "sigmoid":
        return tensor.sigmoid();
      case "tanh":
        return tensor.tanh();
      case "relu":
        return tensor.relu();
      default:
        return tensor;
    }
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

  predict(inputArray) {
    const inputs = tf.tensor2d([inputArray]);
    const hidden = this._activate(
      inputs.matMul(this.weights_ih).add(this.bias_h),
      this.activationHidden
    );
    const output = this._activate(
      hidden.matMul(this.weights_ho).add(this.bias_o),
      this.activationOutput
    );
    const result = output.dataSync(); // 🔥 get result BEFORE tidy
    inputs.dispose();
    hidden.dispose();
    output.dispose();
    return result;
  }

  train(input_array, target_array) {
    tf.tidy(() => {
      const inputs = tf.tensor2d([input_array]);
      const targets = tf.tensor2d([target_array]);

      const hidden = this._activate(
        inputs.matMul(this.weights_ih).add(this.bias_h),
        this.activationHidden
      );
      const outputs = this._activate(
        hidden.matMul(this.weights_ho).add(this.bias_o),
        this.activationOutput
      );

      const output_errors = targets.sub(outputs);

      const gradients = outputs
        .mul(tf.scalar(1).sub(outputs))
        .mul(output_errors)
        .mul(this.learningRate);

      const weights_ho_deltas = hidden.transpose().matMul(gradients);
      this.weights_ho.assign(this.weights_ho.add(weights_ho_deltas));
      this.bias_o.assign(this.bias_o.add(gradients.mean(0)));

      const hidden_errors = output_errors.matMul(this.weights_ho.transpose());
      const hidden_gradient = hidden
        .mul(tf.scalar(1).sub(hidden))
        .mul(hidden_errors)
        .mul(this.learningRate);

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
        .mul(this.learningRate);

      const weights_ho_deltas = hidden.transpose().matMul(gradients);
      this.weights_ho.assign(this.weights_ho.add(weights_ho_deltas));
      this.bias_o.assign(this.bias_o.add(gradients.mean(0)));

      const hidden_errors = output_errors.matMul(this.weights_ho.transpose());
      const hidden_gradient = hidden
        .mul(tf.scalar(1).sub(hidden))
        .mul(hidden_errors)
        .mul(this.learningRate);

      const weights_ih_deltas = inputs.transpose().matMul(hidden_gradient);
      this.weights_ih.assign(this.weights_ih.add(weights_ih_deltas));
      this.bias_h.assign(this.bias_h.add(hidden_gradient.mean(0)));
    });
  }
}

export default NeuralNetwork;
