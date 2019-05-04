import * as math from 'mathjs';

export class TwoLayerNeuralNetwork {
    private weight1: Array<number[]>;
    private weight2: Array<number[]>;
    private outputPredictions: Array<number[]>;
    private layer1;

    constructor(private inputs: Array<number[]>, private expectOutput: Array<number[]>) {
        this.weight1 = this.create2DArray(inputs[1].length, 4);
        this.weight2 = this.create2DArray(4, 1);
        this.outputPredictions = Array(expectOutput.length).fill(0, 0);
    }

    private create2DArray(x: number, y: number): Array<number[]> {
        return Array(x).fill(0, 0).map(() => {
            return Array(y).fill(0, 0).map(() => Math.random());
        });
    }

    public feedForward(): void {
        this.layer1 = (math.multiply(this.inputs, this.weight1) as Array<number[]>).map(row => {
            return row.map((element) => {
                return this.sigmoid(element);
            })
        });
        this.outputPredictions = (math.multiply(this.layer1, this.weight2) as any[]).map(x => [this.sigmoid(x[0])]);
    }

    public backPropagation() {
        const outputDifference = math.subtract(this.expectOutput, this.outputPredictions) as Array<number[]>;
        const derivativePrediction =  this.outputPredictions.map(x => [this.derivativeSigmoid(x[0])]);
        const errorMarginHiddenLayer = outputDifference.map((x, index) => [x[0] * derivativePrediction[index][0]]);
        const hiddenLayerBackPropagation = math.multiply(math.transpose(this.layer1), errorMarginHiddenLayer);
        this.weight2 = math.add(this.weight2, hiddenLayerBackPropagation) as any;


        const inputDifference = math.multiply(errorMarginHiddenLayer, math.transpose(this.weight2)) as Array<number[]>;
        const layer1Corrections = this.layer1.map(row => {
            return row.map(element => this.derivativeSigmoid(element))
        });

        const inputCorrections = layer1Corrections.map((row, rowIndex) => {
            return row.map((element, columnIndex) => {
                return element * inputDifference[rowIndex][columnIndex]
            });
        }) as Array<number[]>;
        const inputBackPropagation = math.multiply(math.transpose(this.inputs), inputCorrections);
        this.weight1 = math.add(this.weight1, inputBackPropagation) as any;
        return this.outputPredictions;
    }

    private derivativeSigmoid(x) {
        return x * (1.0 - x);
    }

    private sigmoid(x) {
        return 1.0/(1+ Math.exp(-x));
    }
}
