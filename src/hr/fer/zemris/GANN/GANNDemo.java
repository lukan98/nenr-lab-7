package hr.fer.zemris.GANN;

public class GANNDemo {

	public static void main(String[] args) {
		sixthProblem();
	}
	
	public static void fourthProblem() {
		Dataset data = new Dataset("/Users/lukanamacinski/FER-workspace/NENR-workspace/lab7/zad7-dataset.txt");

		NeuralNetwork network = new NeuralNetwork(2,8,3);

        GeneticAlgorithm ga = new GeneticAlgorithm(network, data, 2000000);
        ga.execute(true);
        System.out.println("Greška pronađene 2x8x3 mreže: "+network.calcError(data, ga.getBestIndividual()));
        
        int correctCount = network.classify(ga.getBestIndividual(), data);
        
        System.out.println("\nTočnost modela: "+((double)(correctCount/data.getNoOfDatapoints())*100)+"%");
        
        System.out.println("Parametri:");
        double[] parameters = ga.getBestIndividual();
        for (int i=0; i<parameters.length; i++) {
        	System.out.println(parameters[i]);
        }
	}
	
	public static void fifthProblem() {
		Dataset data = new Dataset("/Users/lukanamacinski/FER-workspace/NENR-workspace/lab7/zad7-dataset.txt");

		NeuralNetwork network = new NeuralNetwork(2,8,4,3);

        GeneticAlgorithm ga = new GeneticAlgorithm(network, data, 2000000);
        ga.execute(true);
        System.out.println("Greška pronađene 2x8x4x3 mreže: "+network.calcError(data, ga.getBestIndividual()));
        
        int correctCount = network.classify(ga.getBestIndividual(), data);
        
        System.out.println("\nTočnost modela: "+((double)(correctCount/data.getNoOfDatapoints())*100)+"%");
        
        System.out.println("Parametri:");
        double[] parameters = ga.getBestIndividual();
        for (int i=0; i<parameters.length; i++) {
        	System.out.println(parameters[i]);
        }
	}
	
	public static void sixthProblem() {
		Dataset data = new Dataset("/Users/lukanamacinski/FER-workspace/NENR-workspace/lab7/zad7-dataset.txt");

		NeuralNetwork network = new NeuralNetwork(2,6,4,3);

        GeneticAlgorithm ga = new GeneticAlgorithm(network, data, 2000000);
        ga.execute(true);
        System.out.println("Greška pronađene 2x6x4x3 mreže: "+network.calcError(data, ga.getBestIndividual()));
        
        int correctCount = network.classify(ga.getBestIndividual(), data);
        
        System.out.println("\nTočnost modela: "+((correctCount/(double)data.getNoOfDatapoints())*100)+"%");
        
        System.out.println("Parametri:");
        double[] parameters = ga.getBestIndividual();
        for (int i=0; i<parameters.length; i++) {
        	System.out.println(parameters[i]);
        }
	}

}
