package hr.fer.zemris.GANN;

import java.util.Random;

public class GeneticAlgorithm {

	private static final double EPSILON = 10e-7;
	private static final int POPULATION_SIZE = 90;
	private static final double MUTATION1_PROBABILITY = 0.05;
	private static final double MUTATION2_PROBABILITY = 0.05;
	private static final double SIGMA1 = 1;
	private static final double SIGMA2 = 0.5;
	private static final double SIGMA3 = 1;
	private static final double T1 = 1;
	private static final double T2 = 1;
	private static final double T3 = 1;
	private static final double T = T1 + T2 + T3;

	private NeuralNetwork network;
	private Dataset data;
	private double[][] parameterPopulation;
	private double[] errors;
	private int maxIterations;
	private int bestIndividualIndex;

	public GeneticAlgorithm(NeuralNetwork network, Dataset data, int maxIterations) {

		this.network = network;
		this.data = data;

		this.parameterPopulation = new double[POPULATION_SIZE][network.noOfParameters()];
		this.errors = new double[POPULATION_SIZE];
		this.maxIterations = maxIterations;

		Random rand = new Random();

		for (int i = 0; i < POPULATION_SIZE; i++)
			for (int j = 0; j < network.noOfParameters(); j++)
				this.parameterPopulation[i][j] = rand.nextGaussian();

		getBestIndividualIndex();

	}

	private void calcErrors() {
		for (int i = 0; i < POPULATION_SIZE; i++) {
			double mse = this.network.calcError(this.data, parameterPopulation[i]);
			errors[i] = mse;
		}
	}

	private int getBestIndividualIndex() {
		int bestErrorIndex = 0;

		for (int i = 0; i < POPULATION_SIZE; i++) {
			double mse = this.network.calcError(this.data, parameterPopulation[i]);
			errors[i] = mse;

			if (mse < errors[bestErrorIndex])
				bestErrorIndex = i;

		}

		this.bestIndividualIndex = bestErrorIndex;

		return bestErrorIndex;
	}

	public double[] getBestIndividual() {
		return parameterPopulation[bestIndividualIndex];
	}

	public void execute(boolean verbose) {

		for (int i = 0; i < this.maxIterations; i++) {
			if (i == 0)
				this.calcErrors();

			if (createGeneration()<EPSILON)
				break;
		}

	}

	private double createGeneration() {

		Random rand = new Random();

		int indexA = -1;
		int indexB = -1;
		int indexC = -1;

		indexA = rand.nextInt(POPULATION_SIZE);
		do {
			indexB = rand.nextInt(POPULATION_SIZE);
		} while (indexA == indexB);
		do {
			indexC = rand.nextInt(POPULATION_SIZE);
		} while (indexC == indexB && indexC == indexA);

		double[] individualA = this.parameterPopulation[indexA];
		double[] individualB = this.parameterPopulation[indexB];
		double[] individualC = this.parameterPopulation[indexC];

		double mseA = this.errors[indexA];
		double mseB = this.errors[indexB];
		double mseC = this.errors[indexC];

		double[] parentA, parentB;
		int worstIndex;

		if (mseA > mseB && mseA > mseC) {
			parentA = individualB;
			parentB = individualC;
			worstIndex = indexA;
		} else if (mseB > mseA && mseB > mseC) {
			parentA = individualA;
			parentB = individualC;
			worstIndex = indexB;
		} else {
			parentA = individualA;
			parentB = individualB;
			worstIndex = indexC;
		}

		double[] child;

		int crossoverIndex = new Random().nextInt(3);

		if (crossoverIndex == 0)
			child = arithmeticCrossover(parentA, parentB);
		else if (crossoverIndex == 1)
			child = uniformCrossover(parentA, parentB);
		else
			child = onePointCrossover(parentA, parentB);

		double v1, v2, randDouble;

		v1 = T1 / T;
		v2 = T2 / T;

		randDouble = rand.nextDouble();

		if (randDouble < v1)
			mutator1(child, SIGMA1);
		else if (randDouble < v1 + v2)
			mutator1(child, SIGMA2);
		else
			mutator2(child, SIGMA3);

		double childMSE = this.network.calcError(this.data, child);

		if (childMSE <= this.errors[worstIndex]) {
			this.parameterPopulation[worstIndex] = child;
			this.errors[worstIndex] = childMSE;
		}
		
		return this.errors[worstIndex];

	}

	private void mutator1(double[] individual, double sigma) {
		Random rand = new Random();

		for (int i = 0; i < individual.length; i++) {
			if (rand.nextDouble() < MUTATION1_PROBABILITY)
				individual[i] += rand.nextGaussian() * sigma;
		}

	}

	private void mutator2(double[] individual, double sigma) {
		Random rand = new Random();

		for (int i = 0; i < individual.length; i++) {
			if (rand.nextDouble() < MUTATION2_PROBABILITY)
				individual[i] = rand.nextGaussian() * sigma;
		}

	}

	private double[] arithmeticCrossover(double[] parentA, double[] parentB) {
		double[] child = new double[parentA.length];

		double randDouble = new Random().nextDouble();
		for (int i = 0; i < child.length; i++) {
			child[i] = randDouble * parentA[i] + (1 - randDouble) * parentB[i];
		}

		return child;
	}

	private double[] uniformCrossover(double[] parentA, double[] parentB) {
		double[] child = new double[parentA.length];

		Random rand = new Random();
		for (int i = 0; i < child.length; i++) {
			if (rand.nextDouble() > 0.5)
				child[i] = parentA[i];
			else
				child[i] = parentB[i];
		}

		return child;
	}

	private double[] onePointCrossover(double[] parentA, double[] parentB) {
		double[] child = new double[parentA.length];

		int randIndex = new Random().nextInt(parentA.length);
		for (int i = 0; i < child.length; i++) {
			if (i < randIndex)
				child[i] = parentA[i];
			else
				child[i] = parentB[i];
		}

		return child;
	}

	public void printPop() {
		for (int i = 0; i < POPULATION_SIZE; i++) {
			System.out.println(i + ": " + errors[i]);
		}

		System.out.println("Best error: " + errors[this.bestIndividualIndex]);
	}

	public void printBest() {
		for (int i = 0; i < network.noOfParameters(); i++) {
			System.out.println(this.parameterPopulation[this.bestIndividualIndex][i]);
		}
	}

}
