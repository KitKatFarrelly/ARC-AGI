import os
import argparse
import json
from enum import Enum
import numpy as np

class transform_type(Enum):
    identity = 1
    scale = 2
    rotate = 3
    move = 4
    mirror = 5
    change_color = 6
    invalid = 7

#runs a single transformation on a matrix and returns an output matrix with that transform
def run_transform_on_matrix(input_matrix, transform):
    output_matrix = np.array()
    if(transform is transform_type.identity):
        output_matrix = input_matrix
    elif(transform is transform_type.scale):
        pass
    elif(transform is transform_type.rotate):
        pass
    elif(transform is transform_type.move):
        pass
    elif(transform is transform_type.mirror):
        pass
    elif(transform is transform_type.change_color):
        pass
    else:
        #perform identity transform on invalid transforms
        output_matrix = input_matrix
    return output_matrix

# calculates average difference between input matrix and output matrix
# currently only calculates if input and output matrix are the same size
def calculate_loss(input_matrix, output_matrix) -> float:
    if(input_matrix.shape[0] is not output_matrix.shape[0] or input_matrix.shape[1] is not output_matrix.shape[1]):
        return -1.0 # negative loss means loss could not be calculated for these matrices
    loss = 0.0
    for x in range(input_matrix.shape[0]):
        for y in range(input_matrix.shape[1]):
            if(input_matrix[x][y] is not output_matrix[x][y]):
                loss = loss + 1.0
    loss = loss / (input_matrix.shape[0] * input_matrix.shape[1])
    return loss

#returns list of features extracted via dfs in a matrix.
def find_features_in_matrix(input_matrix):
    pass

#in cases where the set of transforms to solve is dependent on some branching feature,
#this function inputs a matrix and branch set and outputs which element in the set
#which the matrix falls under. If it falls under no element in the set, then a
#new element is created in the branch set and returned.
def run_branch_on_matrix(input_matrix, branch_type):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("inputfile", help="input path to the test file")

args = parser.parse_args()
input_file = os.path.abspath(args.inputfile)

with open(input_file, 'r') as file:
    data = json.load(file)

# Split test and training data
test = data["test"]
train = data["train"]
training_len = len(train)
test_len = len(test)

# Divide Training data into a list of examples
training_example_list = []
for i in train:
    training_example_list.append(i)

test_example = test[0]

# Print the data
print(f"training data length: {training_len}")
for j in range(training_len):
    print(f"training data {j}: \n{training_example_list[j]}")
print(f"test data: \n{test_example}")

# parse input and output arrays per example
training_input = {}
training_output = {}
for j in range(training_len):
    training_input[j] = np.array((training_example_list[j])["input"])
    training_output[j] = np.array((training_example_list[j])["output"])
test_input = np.array((training_example_list[0])["input"])
test_output = np.array((training_example_list[0])["output"])

# okay, now the only thing to do is to convert the inputs into an output using a single method...
