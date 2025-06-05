function NeuralNetwork(numInputs, numHidden, numOutputs) {
    this.input_Nodes = numInputs;
    this.hidden_Nodes = numHidden;
    this.output_Nodes = numOutputs;

    this.weights_ih = new Matrix(this.hidden_Nodes, this.input_Nodes);
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);

    this.weights_ih.randomize();
    this.weights_ho.randomize();
}

export default NeuralNetwork;