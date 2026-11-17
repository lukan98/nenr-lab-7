# Genetic Algorithm Trained Neural Network (NENR Lab 7)

A Java project that trains a feedforward neural network with a genetic algorithm instead of gradient descent. It was written for the **NENR (Fuzzy, Evolutionary and Neurocomputing)** course at FER, University of Zagreb.

## Problem

The network classifies two-dimensional points into three classes. Its architecture is configurable, for example:

- `2 → 8 → 3`
- `2 → 8 → 4 → 3`
- `2 → 6 → 4 → 3`

A real-valued genetic algorithm optimizes all network weights and biases by minimizing the mean squared error on the training dataset.

## Input format

The dataset is a whitespace-separated text file with five columns:

```text
x y class0 class1 class2
```

The last three columns are one-hot class labels.

## Requirements

- Java JDK 11 or newer
- Eclipse is optional; the repository includes Eclipse project metadata

## Build and run

From the project root:

```bash
mkdir -p bin
find src -name "*.java" -print0 | xargs -0 javac -d bin
java -cp bin hr.fer.zemris.GANN.GANNDemo
```

`GANNDemo` calls `sixthProblem()` by default. Change the call in `main()` to run `fourthProblem()` or `fifthProblem()` for different network architectures.

## Notes

`GANNDemo.java` contains a hardcoded absolute path to the original dataset (`zad7-dataset.txt`). The dataset is **not included** in this repository, so edit the path to point to your own file before running.

## Project structure

```text
src/hr/fer/zemris/GANN/
├── Dataset.java          # dataset loading and class utilities
├── NeuralNetwork.java    # network evaluation and classification
├── GeneticAlgorithm.java # real-valued GA over network parameters
└── GANNDemo.java         # experiment entry point
```
