import numpy as np
import math
import pandas as pd

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float) or isinstance(value,np.integer)

def same(L1,L2):
	return ( len(L1.intersection(L2)) )

def different(L1,L2):
	return ( len(L1.symmetric_difference(L2)) )

def cardinality(L1,L2):
	return (len(L1.union(L2)))

def similarity(L1,L2):
	L1 = set(L1)
	L2 = set(L2)
	return (((same(L1,L2)/cardinality(L1,L2)) - (different(L1,L2)/cardinality(L1,L2)) + 1) / 2)

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1,1 << x):
        yield frozenset({ss for mask, ss in zip(masks, s) if i & mask})

def build_similarity_table(data):
	flat_list = []
	for subset in data.iloc[:,-1]:
	    for item in subset:
	        flat_list.append(item)

	global L
	L = list(powerset(list(set(flat_list))))

	w, h = len(L), len(L)
	similarity_table = [[0 for x in range(w)] for y in range(h)] 

	for i,j in enumerate(L):
		for l, k in enumerate(L):
			similarity_table[i][l] = similarity(j,k)

	return similarity_table

class TreeNode():

	def __init__(self, is_leaf = False, label = None, split_on_feature = None):
		self.is_leaf = is_leaf
		self.label = label
		self.split_on_feature = split_on_feature
		self.children = dict()

	def __repr__(self):
		
		if not self.is_leaf:
			message = " Internal node split on feature " + str(self.split_on_feature)
		else:
			message = " Leaf node with label " + str(self.label) + "."
		return message

class DecisionTreeMVMC():

	# Initialize the decision tree with the user-supplied parameters.
	def __init__(self,delta,num_of_intervals,minsup,mindiff,minqty):
		self.delta = delta
		self.num_of_intervals = num_of_intervals
		self.minsup = minsup
		self.mindiff = mindiff
		self.minqty = minqty

	# Fit a tree on the data.
	def fit(self,data,features,target):
		self.features = features
		self.target = target

		self.similarity_table = build_similarity_table(data)

		self.root = self.recursive_build_tree(data = data,features = features,target = target)

	def split_intervals(self,values):
		values = np.array(values)
		generated_intervals = []
		b1 = values[0]
		for i in range(1,self.num_of_intervals + 1):
			if (i == 1):
				left = b1 - 2*self.delta
			else:
				left = right
			index = (i/(self.num_of_intervals))
			right = np.percentile(values,index*100)
			if (left != right):
				generated_intervals.append((left + self.delta, right))
		return generated_intervals

	def set_similarity(self,label):
		if (len(label) == 1):
			return 1

		if (len(label) == 0):
			return 0

		similarity_score = 0

		m = len(label)
		denominator = (m - 1)/2

		for i,j in enumerate(label):
			for k,l in enumerate(label[i+1:]):
				# print(L.index(j),L.index(l))
				similarity_score += self.similarity_table[L.index(j)][L.index(l)]

		return (similarity_score)/(m*denominator)

	def compute_weighted_similarity_numerical(self,attribute,data,intervals):
		
		weighted_similarity = 0
		
		count = len(data.index)

		for interval in intervals: 
			records_within_interval_df = data[(data.iloc[:,attribute] >= interval[0]) & (data.iloc[:,attribute] <= interval[1])]
			label = records_within_interval_df.iloc[:,-1].apply(lambda x : frozenset(x)).tolist()

			setSimiltarity = self.set_similarity(label)

			set_similtarity_score = setSimiltarity*(len(label)/count)

			weighted_similarity += set_similtarity_score

		return weighted_similarity,intervals

	def compute_weighted_similarity_categorical(self,attribute,data):
	
		values = data.iloc[:,attribute]

		values = values.tolist()
		
		weighted_similarity = 0

		count = 0
		unique_values = set()

		for value in values:
			if isinstance(value,set) or isinstance(value,list): 
				count += len(value)
			else:
				count += 1
			
			for elem in value:
				unique_values.add(elem)

		for value in unique_values:
			label = [y for x,y in zip(data.iloc[:,attribute],data.iloc[:,-1]) if value in x]

			setSimiltarity = self.set_similarity(label)

			set_similtarity_score = setSimiltarity*(len(label)/count)

			weighted_similarity += set_similtarity_score

		return weighted_similarity,unique_values

	def find_best_split(self,data,features,target):

		""" Find best split on the data and return the best splitting attribute. 
		If the best attribute is of type numerical, return their intervals aswell.
		If the best attribute is of type categorical, return their unique values.
		"""
		n_features = len(features)
		weighted_similarity = 0
		best_weighted_similarity = 0
		best_feature = 0

		print(features)

		for col in features:
			if (is_numeric(data.iloc[0,col])): #Find if the column in the dataframe is numerical
				data = data.sort_values(data.columns[col])
				intervals = [(100,224),(225,349),(350,474),(475,599),(600,724),(725,849),(850,974),(975,1099),(1100,1224),(1225,1350)] #self.split_intervals(data.iloc[:,col])
				
				weighted_similarity,intervals = self.compute_weighted_similarity_numerical(col,data,intervals)
			else:
				weighted_similarity,unique_values = self.compute_weighted_similarity_categorical(col, data)

			if (weighted_similarity >= best_weighted_similarity):
				best_weighted_similarity, best_feature = weighted_similarity, col
		
		if is_numeric(data.iloc[0,best_feature]):	
			return best_weighted_similarity,best_feature,intervals
		else:
			return best_weighted_similarity,best_feature,unique_values

	def partition_for_numerical(self,data,attribute,intervals):
		branches = []
		for interval in intervals:
			selected_rows = data.loc[(data.iloc[:,attribute] >= interval[0]) & (data.iloc[:,attribute] <= interval[1])]
			branches.append(selected_rows.values.tolist())
		return branches

	def is_stop_node(self,branch,features):
		# Check if branch is empty, handle it
		unique_labels = set()
		
		for row in branch:
			for labels in row[-1]:
				unique_labels.add(labels)

		support = {}
		for label in unique_labels:
			rows_with_label = 0
			for row in branch:
				if(label in row[-1]):
					rows_with_label += 1
			support[label] = (rows_with_label/len(branch))*100

		small = set()
		large = set()
		for label in unique_labels:
			if (support[label] >= self.minsup):
				large.add(label)
			else:
				small.add(label)

		minimum_support = math.inf
		for label in large:
			if (support[label] < minimum_support):
				minimum_support = support[label]

		if (len(large) == 0):
			minimum_support = 0

		maximum_support = 0
		for label in small:
			if (support[label] > maximum_support):
				maximum_support = support[label]

		difference = minimum_support - maximum_support

		if difference >= self.mindiff: # Clear
			return True,large # Clear node, and can be stopped
		else: # Unclear 
			if (len(features) == 0): # If all attributes have been used in the path from root-down to Current Node.
				if (len(large) != 0):
					return True,large
				else:
					list_of_support = sorted(support.items(), key=lambda x: x[1])
					return True,list_of_support[0][0] 
			elif (len(branch) < self.minqty) : # If number of data records is smaller than minqty
				if (len(large) != 0):
					return True,large
				else:
					list_of_support = sorted(support.items(), key=lambda x: x[1])
					print("251")
					print(list_of_support)
					return True,list_of_support[0][0]
			else:
				return False,None

	def recursive_build_tree(self,data,features,target):
		
		print("262")
		data = pd.DataFrame(data)
		print(data)

		best_weighted_similarity,best_feature, _ = self.find_best_split(data,features,target)

		node = TreeNode(is_leaf = False,label = None,split_on_feature = best_feature)

		print(data.iloc[0,best_feature])
		if (is_numeric(data.iloc[0,best_feature])):	 
			intervals = [(100,224),(225,349),(350,474),(475,599),(600,724),(725,849),(850,974),(975,1099),(1100,1224),(1225,1350)] #self.split_intervals(data.iloc[:,best_feature])
			print("intervals:")
			print(intervals)
			branches = self.partition_for_numerical(data,best_feature,intervals)
			
			for i in branches:
				print(i)

			features.remove(best_feature) # To avoid splitting on the same feature again

			print(features)

			print("285")
			print(branches)
			
			branches = [x for x in branches if x]

			print("291")
			print(branches)

			for i,branch in enumerate(branches):

				stop,label_set = self.is_stop_node(branch, features)
				print(stop,label_set)

				if (not stop):
					child_node = self.recursive_build_tree(branch,features,target)
					node.children[i] = child_node
				else:
					# Assign the label set to that branch
					node = TreeNode(is_leaf = True,label = label_set)
		# else:
		# 	unique_values = _
		# 	print("303")
		# 	print(unique_values)
		# 	branches = self.partition_for_categorical(data,best_feature,unique_values,features)
		# 	print("306")
		# 	print(branches)
		# 	features.remove(best_feature) # To avoid splitting on the same feature again

		# 	for branch in branches:

		# 		stop,label_set = self.is_stop_node(branch,features)

		# 		if (not stop):
		# 			child_node = self.recursive_build_tree(branch,features,target)
		# 			node.children[branch] = child_node
		# 			print(node)
		# 		else:
		# 			# Assign the label set to that branch
		# 			leaf = TreeNode(is_leaf = True,label = label_set)
		# 			print(leaf)

		
		return node
		