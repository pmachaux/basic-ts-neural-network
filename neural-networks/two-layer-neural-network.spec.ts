import {TwoLayerNeuralNetwork} from "./two-layer-neural-network";


describe('Neural network', () => {

  let neuralNetwork: TwoLayerNeuralNetwork;
  let expectedOutputs;
  beforeEach(() => {
    const inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]];
    expectedOutputs = [[0], [1], [1], [0]];

    neuralNetwork = new TwoLayerNeuralNetwork(inputs, expectedOutputs);
  });

  it('should learn', () => {
    let result;
      for(let i =0; i < 10500; i++) {
      neuralNetwork.feedForward();
      result = neuralNetwork.backPropagation();
      }
    expect(Math.abs(result[0][0] - expectedOutputs[0][0])).toBeLessThan(0.1);
    expect(result[1][0] - expectedOutputs[1][0]).toBeLessThan(0.1);
    expect(result[2][0] - expectedOutputs[2][0]).toBeLessThan(0.1);
    expect(Math.abs(result[3][0] - expectedOutputs[3][0])).toBeLessThan(0.1);
    console.log(result);
  });

});
