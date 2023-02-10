import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class DecisionTree {
	private static Random rand = new Random();
	private static List<String[]> properties = new ArrayList<String[]>();

	public static void main(String[] args) throws IOException {		
		//Read in command line arguments
		int iVal=10, lVal=100, tVal=20;
		boolean isPrinting = false;
		try {
			for (int i = 0; i < args.length; i++) {
				if (args[i].equals("-i")) {
					iVal = Integer.parseInt(args[++i]);
				}
				else if (args[i].equals("-l")) {
					lVal = Integer.parseInt(args[++i]);
				}
				else if (args[i].equals("-t")) {
					tVal = Integer.parseInt(args[++i]);
				}
				else if (args[i].equals("-p")) {
					isPrinting = true;
				} else {
					throw new Exception();
				}
			}
		} catch (NumberFormatException e) {
			System.err.println("ERROR: VAL WAS NOT AN INTEGER: " + e.getMessage());
		} catch (Exception e) {
			System.err.println("ERROR WHILE PARSING COMMAND LINE ARGS: " + e.getMessage());
		}
		
		// Load the data from the files
		List<String> data = new ArrayList<String>();
    	readInFiles(data);
    	
    	// Select training data and build Decision Tree
    	Set<String> trainingData = new HashSet<String>();
    	Set<String> testingData = new HashSet<String>(data);
    	TreeNode tree = null;
    	
    	System.out.println("(Results averaged across " + tVal + " trials)");
    	System.out.println("TrainSize\tTrainAcc\tTestAcc");
    	//for each training group size
    	for (int k = 1; iVal * k <= lVal; k++) {
    		double avgTrainingAccuracy = 0, avgTestingAccuracy = 0;
    		int s = iVal * k; // size of the training set
    		//for each trial
    		for (int t = 0; t < tVal; t++) {
    			// Generate a random training set from the data and build the tree
    			selectTrainingData(data, trainingData, testingData, s);
	    		tree = LearnDecisionTree(trainingData, getAttributes(), null);
	    		
	    		// Test Decision tree with Training data
				double trainingAccuracy = testTreeWithTrainingData(trainingData, tree);
	    		avgTrainingAccuracy += trainingAccuracy;
	    		
	    		// Test decision tree with test data (total data - training data)
	    		double testingAccuracy = testTreeWithTestingData(testingData, tree); 	
	    		avgTestingAccuracy += testingAccuracy;
	    		
    		}
    		avgTrainingAccuracy = (avgTrainingAccuracy/((double) tVal));
    		avgTestingAccuracy = (avgTestingAccuracy/((double) tVal));
    		System.out.println("" + s + "\t\t" + avgTrainingAccuracy + "\t\t" + avgTestingAccuracy);
    	}	
    	if (isPrinting && tree != null) {
    		tree.printTree();
    	}
	}
	
	/***
	 * Read in the input files containg the mushroom data and the properties describing each attribute
	 * @param data - the data set to be filled form mushroom_data.txt
	 * @throws FileNotFoundException - if either file is not found
	 * @throws IOException - if there is a problem reading in either of the files
	 */
	private static void readInFiles(List<String> data) throws FileNotFoundException, IOException {
		try {
			File dataFile=new File("mushroom_data.txt");    //creates a new file instance  
			FileReader fr=new FileReader(dataFile);   //reads the file  
			BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream  
			String line;  
			while((line=br.readLine())!=null) {  
				data.add(line.replaceAll(" ", ""));      //appends line to string buffer  
			}
			fr.close();    
			
			File propFile=new File("properties.txt");    //creates a new file instance  
			fr = new FileReader(propFile);   //reads the file  
			br = new BufferedReader(fr);  //creates a buffering character input stream  
			while ((line=br.readLine()) != null) {  
				line = line.substring(line.indexOf(':')+1, line.length());
				properties.add(line.split(" "));
			}
			fr.close();    
		} catch (Exception e) {
			System.err.println("ERROR WHILE READING IN FILES: " + e.getMessage());
		}
	}

	/***
	 * Function to create the set of attributes, which in this case are just int values 0-21
	 * @return the set of attributes
	 */
	private static Set<Integer> getAttributes() {
		Set<Integer> attributes = new HashSet<Integer>();
		for (int i = 0; i < 22 /*number of attributes*/; i++) {
			attributes.add(i);
		}
		return attributes;
	}

	/***
	 * Test the Decision Tree with the training data it was built with
	 * @param trainingData - the data it was built with
	 * @param tree - the Decision Tree to test
	 * @return training accuracy of the tree
	 */
	private static double testTreeWithTrainingData(Set<String> trainingData, TreeNode tree) {
		int correctCount = 0;
		TreeNode current = tree;
		for (String example : trainingData) {
			while (!current.isLeaf()) {
				current = current.getBranch(example.charAt(current.getRoot()));
			}
			correctCount += current.getLeaf() == example.charAt(example.length()-1) ? 1 : 0;
			current = tree;
		}
		double trainingAccuracy = ((double)correctCount)/((double)trainingData.size());
		return trainingAccuracy;
	}
	
	/***
	 * Test the Decision Tree with data it was not trained on
	 * @param testingData - new data to test the tree with
	 * @param tree - the Decision Tree to test
	 * @return testing accuracy of the tree
	 */
	private static double testTreeWithTestingData(Set<String> testingData, TreeNode tree) {
		int correctCount;
		TreeNode current;
		correctCount = 0;
		current = tree;
		TreeNode working;
		for (String example : testingData) {
			while(!current.isLeaf()) {
				working = current.getBranch(example.charAt(current.getRoot()));
				if (working != null) {
					current = working;
				} else {
					break;
				}
			}
			correctCount += current.getLeaf() == example.charAt(example.length()-1) ? 1 : 0;
			current = tree;
		}
		
		double testingAccuracy = ((double)correctCount)/((double)testingData.size());
		return testingAccuracy;
	}

	/***
	 * Select s examples from the data to be used as training data
	 * @param data - the data to choose from
	 * @param trainingData - the set of selected data examples
	 * @param testingData - the leftover data to be used to test the tree
	 * @param s - the number of examples to grab from data to make the training set
	 */
	private static void selectTrainingData(List<String> data, Set<String> trainingData, Set<String> testingData, int s) {
		String element;
		while (trainingData.size() < s) {
			element = data.get(rand.nextInt(data.size()));
			trainingData.add(element);
		}
		testingData.removeAll(trainingData);
	}

	//------------------------The decision tree building algorithm and all of its helper functions---------------------------//
	
	/***
	 * This algorithm builds a decision tree following the algorithm specified in 
	 * Russell and Norvig (Figure 19.5, page 660, in fourth edition).
	 * @param examples - the training data to use to build the tree
	 * @param attributes - the set of attributes to look at and split upon
	 * @param parentExamples - the set of examples from the previous recursive call (empty in first call)
	 * @return the Decision Tree
	 */
	private static TreeNode LearnDecisionTree(Set<String> examples, Set<Integer> attributes, Set<String> parentExamples) {
		Character classification;
		if (examples.isEmpty()) {
			return PluralityValue(parentExamples);
		} else if ((classification = allExamplesHaveTheSameClassification(examples.toArray())) != null) {
			return new TreeNode(classification);
		} else if (attributes.isEmpty()) {
			return PluralityValue(examples);
		} else {
			int A = argmax(attributes, examples);
			TreeNode tree = new TreeNode(A);
			for (char v : valuesOfA(A)) {
				Set<String> exs = getExs(examples, A, v);
				attributes.remove(A); // maybe make a new set?
				TreeNode subtree = LearnDecisionTree(exs, attributes, examples);
				tree.addBranch(v, subtree);
			}
			return tree;
		}
	}
	
	/***
	 * Selects the most common classification, breaking ties randomly
	 * @param examples - the set to select the most common classification form
	 * @return the classification
	 */
	private static TreeNode PluralityValue(Set<String> examples) {
		int edibleCount = 0;
		int poisonCount = 0;
		char c;
		for (String e : examples) {
			c = e.charAt(e.length()-1);
			if (c == 'e') {
				edibleCount++;
			} else if (c == 'p') {
				poisonCount++;
			}
		}
		if (edibleCount > poisonCount) {
			c = 'e';
		} else if (edibleCount < poisonCount) {
			c = 'p';
		} else {
			c = rand.nextDouble() > 0.5 ? 'e' : 'p';
		}
		return new TreeNode(c);
	}
	
	/***
	 * checks if all examples have the same classification
	 * @param examples - the examples to check over
	 * @return the classification if true, null otherwise
	 */
	private static Character allExamplesHaveTheSameClassification(Object[] examples) {
		char first = ((String) examples[0]).charAt(((String) examples[0]).length()-1);
		for (Object e : examples) {
			if (((String) e).charAt(22) != first) {
				return null;
			}
		}
		return first;
	}
	
	/***
	 * Function to calculate the log base 2 of an integer 
	 * @param N - the interger to take log base 2 of
	 * @return log base 2 of N
	 */
    private static double log2(double N) { 
        return (Math.log(N) / Math.log(2));  
    } 
    
    /***
     * Calculate how important a particular attribute is
     * Used to compare with the importance of all other attributes in order to decide which
     * attribute to split on that would result in the most compact and efficient decision tree
     * @param a - the attribute to check
     * @param examples - the set of examples to check attribute a for
     * @return the information gain/importance value calculated for the attribute
     */
	private static double importance(int a, Set<String> examples) {
		int poisonCount = 0;
		Map<Character, List<String>> sets = new HashMap<Character, List<String>>();
		for (String e : examples) {
			char c = e.charAt(a);
			if (sets.get(c) == null) {
				sets.put(c, new LinkedList<String>());
			}
			sets.get(c).add(e);
			if (e.charAt(e.length()-1) == 'p') {
				poisonCount++;
			}
		}
		double entropy = getEntropy(poisonCount, ((double) examples.size()));
		
		//Remainder
		double remainder = 0;
		for (char k : sets.keySet()) {
			int pk = 0;
			for (String e : sets.get(k)) {
				if (e.charAt(e.length()-1) == 'p') {
					pk++;
				}
			}
			double thisEntropy = getEntropy(pk, sets.get(k).size());
			double addToRemainder = (((double)sets.get(k).size())/((double) examples.size())) * thisEntropy;
			remainder += addToRemainder;
		}
		return entropy - remainder;
	}
    
    /***
     * Calculates the entropy of a set of examples
     * @param p - numerator of proportion of the examples that are poisonous 
     * @param pPlusN - the total amount of examples
     * @return the entropy value
     */
    private static double getEntropy(double p, double pPlusN) {
    	double prob = p / pPlusN;
    	if (prob == 0 || prob == 1) return 0;
    	else return -(prob*log2(prob) + (1-prob)*log2(1-prob));
    }
	
    /***
     * Selects the attribute with the highest importance, currently
     * @param attributes - the set of all attributes
     * @param examples - the set to analyze importance on
     * @return the most important attribute at the time, or pseudo-randomly if its all a tie
     */
	private static int argmax(Set<Integer> attributes, Set<String> examples) {
		int A = -1;
		double maxA = Double.MIN_VALUE;
		for (int a : attributes) {
			double imp = importance(a, examples);
			if (imp > maxA) {
				maxA = imp;
				A = a;
			}
		}
		if (A == -1) {
			for(int a : attributes) {
			    return a;
			}
		}
		return A;
	}
	
	/***
	 * Retrieve all possible values for attribute A
	 * @param A - attribute to get values of
	 * @return - set of all possible values for A
	 */
	private static Set<Character> valuesOfA(int A) {
		Set<Character> values = new HashSet<Character>();
		String[] str = properties.get(A);
		for (String s : str) {
			if (!s.equals("")) values.add(s.charAt(0));
		}
		return values;
	}
	
	/***
	 * Get the value of attribute A for data example e
	 * @param e - data example from data set
	 * @param A - attribute to check in the example
	 * @return the value of attribute A in example e
	 */
	private static char getValue(String e, int A) {
		return e.charAt(A);
	}
	
	/***
	 * When creating a new tree node, this function is called to get the new subset of examples to place 
	 * in each particular branch of the tree
	 * @param examples - the examples to get a subset of
	 * @param A - the attribute to check
	 * @param v - the value to check for at attribute A
	 * @return the set of all examples exs in examples examples that have value v for attribute A
	 */
	private static Set<String> getExs(Set<String> examples, int A, char v) {
		Set<String> exs = new HashSet<String>();
		for (String e : examples) {
			if (getValue(e, A) == v) {
				exs.add(e);
			}
		}
		return exs;
	}
}
