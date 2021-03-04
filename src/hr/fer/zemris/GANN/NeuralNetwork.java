package hr.fer.zemris.GANN;

import java.util.Arrays;

public class NeuralNetwork {
	
	private int[] layers;
	private int[] layerIndexes; // indeks prvog neurona pojedinog sloja u ukupnom polju neurona
	private double[] neuronOutputs;
	
	public NeuralNetwork(int ...architecture) {
		this.layers = architecture;
		this.layerIndexes = new int[architecture.length];
		int neuronCounter = 0;
		for(int i=0; i<architecture.length; i++) {
			this.layerIndexes[i] = neuronCounter;
			neuronCounter+=layers[i];
		}
		this.neuronOutputs = new double[neuronCounter];
	}
	
	public int noOfParameters() {
		int parameterCounter = 0;
		
		for (int i=1; i<layers.length; i++) {
			if (i==1)
				parameterCounter += layers[i]*layers[i-1]*2;
			else
				parameterCounter += layers[i]*(layers[i-1]+1);
		}
		
		return parameterCounter;
	}
	
	public double[] calcOutput(double[] input, double[] parameters) {
		if (input.length != layers[0]) throw new IllegalArgumentException("Broj ulaza i broj ulaznih neurona se ne poklapaju!");
		if (parameters.length != this.noOfParameters()) throw new IllegalArgumentException("Neodgovarajući broj parametara za ovu mrežu!");
		
		for (int i=0; i<input.length; i++)
			this.neuronOutputs[i] = input[i];
		
		
		int parameterIndex = 0;

		for (int i=1; i<this.layers.length; i++) {
			int prevStart = this.layerIndexes[i-1];
			int prevEnd = prevStart + this.layers[i-1];
			 
			for (int j=0; j<this.layers[i]; j++) {
				
				double sum = 0;
				double w0 = 0;
				if (i!=1) {
					w0 = parameters[parameterIndex];
					parameterIndex++;
				}
				for (int k=prevStart; k<prevEnd; k++) {
					double x = this.neuronOutputs[k];
					if (i==1) {
						double w = parameters[parameterIndex];
						if(parameters[parameterIndex+1] < 10e-6)
							parameters[parameterIndex+1] = 10e-6;
						double s = parameters[parameterIndex+1];
						parameterIndex += 2;
						sum += Math.abs(x-w)/Math.abs(s);
					} else {
						double w = parameters[parameterIndex];
						parameterIndex++;
						sum += x*w;
					}
				}
				if (i==1)
					this.neuronOutputs[prevEnd+j] = neuron1Function(sum);
				else
					this.neuronOutputs[prevEnd+j] = sigmoid(sum+w0);
			}
			
		}
		
		int outputStart = this.layerIndexes[layerIndexes.length-1];
		int outputSize = this.layers[layers.length-1];
		double[] output = new double[outputSize];
		
		for (int i=0; i<outputSize; i++) {
			output[i] = this.neuronOutputs[outputStart+i];
		}
		
		return output;
	}
	
	public double calcError(Dataset data, double[] parameters) {
		if(parameters.length != this.noOfParameters()) throw new IllegalArgumentException("Neodgovarajući broj parametara!");
		if(data.getDatapoint(0).length != this.layers[0]+this.layers[layers.length-1]) throw new IllegalArgumentException("Mreža nije usklađena sa podatcima (broj inputa/outputa)");
		
		int inputSize = this.layers[0];
		int outputSize = this.layers[layers.length-1];
		
		double totalError = 0;
		
		for (int i=0; i<data.getNoOfDatapoints(); i++) {
			double[] sample = data.getDatapoint(i);
			double[] input = new double [inputSize];
			double[] realOutput = new double [outputSize];
			
			for (int j=0; j<inputSize; j++) {
				input[j] = sample[j];
			}
			for (int j=inputSize; j<inputSize+outputSize; j++) {
				realOutput[j-inputSize] = sample[j];
			}
			
			double[] networkOutput = this.calcOutput(input, parameters);
			double sum = 0;
			
			for (int j=0; j<outputSize; j++)
				sum += Math.pow(networkOutput[j]-realOutput[j], 2);
				
			
			totalError += sum;
		}
		
		return totalError/data.getNoOfDatapoints();
	}
	
	private double neuron1Function(double sum) {
		return 1/(double)(1+sum);
	}
	
	private double sigmoid(double sum) {
		return 1/(double)(1+Math.exp(-sum));
	}

	public int classify(double[] parameters, Dataset data) {
		if (this.noOfParameters()!=parameters.length) throw new IllegalArgumentException("Neodgovarajući broj parametara za ovu mrežu!");
		
		StringBuilder sb = new StringBuilder();
		int correctCount = 0;
		
		int inputSize = this.layers[0];
		int outputSize = this.layers[layers.length-1];
		
		for (int i=0; i<data.getNoOfDatapoints(); i++) {
			double[] sample = data.getDatapoint(i);
			
			double[] input = new double[inputSize];
			double[] trueOutput = new double[outputSize];
			double[] networkOutput = new double[outputSize];
			
			for (int j=0; j<inputSize; j++)
				input[j] = sample[j];
			for (int j=inputSize; j<inputSize+outputSize; j++)
				trueOutput[j-inputSize] = sample[j];
			for (int j=0; j<outputSize; j++)
				networkOutput[j] = this.calcOutput(input, parameters)[j] < 0.5 ? 0 : 1;
			
			if (Arrays.equals(networkOutput, trueOutput))
				correctCount++;
			
			sb.append("Mreža: "+Arrays.toString(networkOutput)+"    |    "+"Stvarnost: "+Arrays.toString(trueOutput)+"    |    "+(Arrays.equals(networkOutput, trueOutput) ? "Točno" : "Netočno")+"\n");
		}
		System.out.print(sb.toString());
		
		return correctCount;
		
	}
	
}
