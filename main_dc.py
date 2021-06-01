from DecisionTreeMVMC import DecisionTreeMVMC
import pandas as pd
import numpy as np

#Parameters that are based on the data, and should be changed as necessary
delta = 0.01
num_of_intervals = 10
minsup = 60
mindiff = 10
minqty = 2

# data = [
# 	['Id-1', {'Arts','Shopping'},15,{'C1','C2'}],
#     ['Id-2', {'Arts','Sports'},17,{'C2','C3'}],
#     ['Id-3', {'Arts'},28,{'C3'}],
#     ['Id-4', {'Shopping','Sports'},15,{'C1','C2','C3'}],
#     ['Id-5', {'Arts','Shopping','Sports'},15,{'C2'}],
#     ['Id-6', {'Shopping','Sports'},25,{'C1'}],
#     ['Id-7', {'Sports'},28,{'C2','C3'}],
#     ['Id-8', {'Arts','Shopping'},18,{'C1','C3'}],
#     ['Id-9', {'Shopping','Sports'},15,{'C1','C2','C3'}],
#     ['Id-10', {'Shopping'},18,{'C2'}],
# ]

data = [
[{'M'},100,{'Female'},{'Arts'},{'C1','C2','C3'}],
[{'S'},880,{'Male'},{'Arts'},{'C2','C3'}],
[{'M'},370,{'Female'},{'Arts','Shopping'},{'C1'}],
[{'D'},1230,{'Male'},{'Sports'},{'C2'}],
[{'S'},910,{'Male'},{'Arts','Sports'},{'C2','C3'}],
[{'S'},770,{'Female'},{'Arts'},{'C1','C2','C3'}],
[{'S'},590,{'Female'},{'Arts','Shopping'},{'C1','C2'}],
[{'D'},1350,{'Male'},{'Shopping'},{'C1','C2','C3'}],
[{'D'},1250,{'Male'},{'Arts','Shopping'},{'C1','C2','C3'}],
[{'S'},1140,{'Male'},{'Arts','Shopping'},{'C1'}],
[{'M'},340,{'Female'},{'Arts','Sports'},{'C1','C3'}],
[{'D'},1300,{'Male'},{'Arts'},{'C1','C2'}],
[{'S'},1090,{'Male'},{'Sports'},{'C3'}],
[{'S'},810,{'Male'},{'Shopping'},{'C1'}],
[{'S'},520,{'Female'},{'Arts','Sports','Shopping'},{'C3'}],
]

# data = [
# [4.8,3.4,1.9,0.2,{"positive"}],
# [5	,3,	1.6,	1.2,	{"positive"}],
# [5,	3.4,	1.6,	0.2,	{"positive"}],
# [5.2,	3.5,	1.5,	0.2,	{"positive"}],
# [5.2,	3.4,	1.4,	0.2,	{"positive"}],
# [4.7,	3.2,	1.6,	0.2,	{"positive"}],
# [4.8,	3.1,	1.6,	0.2,	{"positive"}],
# [5.4,	3.4,	1.5,	0.4,	{"positive"}],
# [7,	3.2,	4.7,	1.4,	{"negative"}],
# [6.4,	3.2,	4.7,	1.5,	{"negative"}],
# [6.9,	3.1,	4.9,	1.5,	{"negative"}],
# [5.5,	2.3,	4,	1.3,	{"negative"}],
# [6.5,	2.8,	4.6,	1.5,	{"negative"}],
# [5.7,	2.8,	4.5,	1.3,	{"negative"}],
# [6.3,	3.3,	4.7,	1.6,	{"negative"}],
# [4.9,	2.4,	3.3,	1, {"negative"}]
# ]


data = pd.DataFrame(data, columns = ['Marital Status','Income','Gender','Hobby', 'Class Label'])

# data = pd.DataFrame(data, columns = ['A','B','C','D','Target'])

tree = DecisionTreeMVMC(delta,num_of_intervals,minsup,mindiff,minqty)
features = [0,1,2,3]
target = 4
tree.fit(data,features,target)
