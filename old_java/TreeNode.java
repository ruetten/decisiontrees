import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

public class TreeNode {
	private static HashMap<Integer, String> attributeMap = new HashMap<Integer, String>();
	
	private int root = -1; // attribute that is being split on at this node
	private Map<Character, TreeNode> branches; // the branches, one for each possible value of the attribute
	char leaf = 0; // 0 if not a leaf, 'e' or 'p' (the classification) if it is
	
	/***
	 * This constructor is only called when creating a non-leaf node
	 * @param root - the attribute that is being split on at this node
	 */
	public TreeNode(int root) {
		populateAttributeMap();
		this.root = root;
		branches = new HashMap<Character, TreeNode>();
	}
	
	/***
	 * This constructor is only called when creating a leaf node
	 * @param leaf - the classification at this leaf
	 */
	public TreeNode(char leaf) {
		populateAttributeMap();
		this.leaf = leaf;
	}
	
	private void populateAttributeMap() {
		attributeMap.put(0, "Cap Shape");
		attributeMap.put(1, "Cap Surface");
		attributeMap.put(2, "Cap Color");
		attributeMap.put(3, "Bruises");
		attributeMap.put(4, "Odor");
		attributeMap.put(5, "Gill Attatchment");
		attributeMap.put(6, "Gill Spacing");
		attributeMap.put(7, "Gill Size");
		attributeMap.put(8, "Gill Color");
		attributeMap.put(9, "Stalk Shape");
		attributeMap.put(10, "Stalk Root");
		attributeMap.put(11, "Stalk surf. above ring");
		attributeMap.put(12, "Stalk surf. below ring");
		attributeMap.put(13, "Stalk color above ring");
		attributeMap.put(14, "Stalk color below ring");
		attributeMap.put(15, "Veil type");
		attributeMap.put(16, "Veil color");
		attributeMap.put(17, "Ring number");
		attributeMap.put(18, "Ring type");
		attributeMap.put(19, "Spore print color");
		attributeMap.put(20, "Population");
		attributeMap.put(21, "Habitat");
	}
	
	/***
	 * @return true if is a leaf node, false otherwise
	 */
	public boolean isLeaf() {
		return leaf != 0;
	}
	
	/**
	 * @return the classification identified at this leaf node
	 */
	public char getLeaf() {
		return leaf;
	}
	
	/***
	 * add a new branch to this particular tree node
	 * @param c - the attribute value the branch is associated with
	 * @param branch - the subtree/branch to add
	 */
	public void addBranch(char c, TreeNode branch) {
		branches.put(c, branch);
	}
	
	/***
	 * @param c - the attribute value the subtree is associated with
	 * @return the subtree
	 */
	public TreeNode getBranch(char c) {
		return branches.get(c);
	}
	
	/***
	 * @return the root value / the attribute that is being split on at this node
	 */
	public int getRoot() {
		return root;
	}
	
	/***
	 * Begins the recursive call to print the tree
	 */
	public void printTree() {
		Map<Integer, Map<Character, String>> valueStrings = new HashMap<Integer, Map<Character, String>>(); 
		
		try {
			File dataFile=new File("value_cheatsheet.txt");    //creates a new file instance  
			FileReader fr=new FileReader(dataFile);   //reads the file  
			BufferedReader br=new BufferedReader(fr);  //creates a buffering character input stream  
			String line;  
			int a = 1;
			while((line=br.readLine())!=null) {  
				for (String vs : line.split(",")) {
					String str = vs.substring(0, vs.indexOf('=')).strip();
					String ch = vs.substring(vs.indexOf('=')+1);
					if (valueStrings.get(a) == null) {
						valueStrings.put(a, new HashMap<Character, String>());
					}
					valueStrings.get(a).put(ch.charAt(0), str);
				}
				a++;
			}
			fr.close();
		} catch (Exception e) {
			System.err.println("ERROR WHILE READING IN FILES: " + e.getMessage());
		}
		
		System.out.println();
		System.out.println("--------------------------------------");
		System.out.println("--- -- -- - Decision  Tree - -- -- ---");
		System.out.println("--------------------------------------");
		this.printTree(1, valueStrings);
	}
	
	/***
	 * The recursive call to print the tree
	 * @param level - the amount of branches deep the current printer is
	 */
	private void printTree(int level, Map<Integer, Map<Character, String>> vs) {
		if (this.isLeaf()) {
			for (int i = 0; i < level-1; i++) {
				System.out.print("   |");
			}
			System.out.print("  ");
			System.out.print("└──");
			System.out.println(leaf == 'p' ? "Poisonous" : "Edible");
		} else {
			for (int i = 0; i < level-1; i++) {
				System.out.print("   |");
			}
			System.out.print("   ");
			System.out.println("Attr" + (root+1) + ": " + attributeMap.get(root));
			for(Character b : branches.keySet()) {
				for (int i = 0; i < level-1; i++) {
					System.out.print("   |");
				}
				System.out.print("   ");
				System.out.print("└──");
				System.out.println(vs.get(root+1).get(b));
				branches.get(b).printTree(level+1, vs);
			}
		}
	}
}