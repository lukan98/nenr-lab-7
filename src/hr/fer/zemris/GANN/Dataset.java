package hr.fer.zemris.GANN;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Dataset {
	
	private ArrayList<double[]> data;
	
	public Dataset(String sourceString) {
		try (BufferedReader br = new BufferedReader(new FileReader(sourceString))) {
			String line;
			this.data = new ArrayList<>();
			
			while((line = br.readLine()) != null) {
				String[] parts = line.split("\\s+");
				
				double[] dataPoint = new double[5];
				
				for (int i=0; i<5; i++)
					dataPoint[i] = Double.parseDouble(parts[i]);
				
				this.data.add(dataPoint);
			}
			
		} catch (IOException e) {
			System.err.println("Could not read from dataset file!");
			e.printStackTrace();
		}
	}
	
	public int getNoOfDatapoints() {
		return data.size();
	}
	
	public double[] getDatapoint(int index) {
		return data.get(index);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		for (double[] dataPoint : data) {
			sb.append("x=").append(dataPoint[0]).append(", y=").append(dataPoint[1])
			  .append(", f(x,y)=").append(dataPoint[2]).append(",").append(dataPoint[3]).append(",").append(dataPoint[4]+"\n");
		}
		
		return sb.toString();
	}
	
	public void printCategory(int index) {
		if (index>2 || index<0) throw new IllegalArgumentException("Index must be from {0, 1, 2}");
		
		StringBuilder sb = new StringBuilder();
		
		for (double[] dataPoint : data) {
			if(dataPoint[2+index] == 1)
				sb.append(dataPoint[0]+" ").append(dataPoint[1]+" ");
		}
		
		System.out.println(sb.toString());
	}

}
