import os
import argparse
import json
import math
from enum import Enum
import numpy as np

class transform_type(Enum):
    identity = 1
    scale = 2
    rotate = 3
    move = 4
    mirror = 5
    change_color = 6
    scale_board = 7
    mask = 8

# runs a single transformation on a matrix and returns an output matrix with that transform
def run_transform_on_matrix(input_matrix, transform, transform_data):
    if(transform is transform_type.identity):
        return np.copy(input_matrix)
    elif(transform is transform_type.scale):
        output_matrix = np.copy(input_matrix)
        for height in range(output_matrix.shape[1]):
            for width in range(output_matrix.shape[0]):
                output_matrix[height, width] = input_matrix[math.floor(height / transform_data), math.floor(width / transform_data)]
        return output_matrix
    elif(transform is transform_type.rotate):
        pass
    elif(transform is transform_type.move):
        pass
    elif(transform is transform_type.mirror):
        pass
    elif(transform is transform_type.change_color):
        pass
    elif(transform is transform_type.scale_board):
        output_matrix = np.zeros(shape=(transform_data[0], transform_data[1]))
        max_copy_width = transform_data[1]
        if input_matrix.shape[1] < max_copy_width:
            max_copy_width = input_matrix.shape[1]
        max_copy_height = transform_data[0]
        if input_matrix.shape[0] < max_copy_height:
            max_copy_height = input_matrix.shape[0]
        output_matrix[0:max_copy_height, 0:max_copy_width] = input_matrix
        return output_matrix
    elif(transform is transform_type.mask):
        output_matrix = np.copy(input_matrix)
        mask_min_height = 0
        while mask_min_height < output_matrix.shape[0]:
            mask_min_width = 0
            max_copy_height = transform_data.shape[0]
            if output_matrix.shape[0] < max_copy_height:
                max_copy_height = output_matrix.shape[0]
            while mask_min_width < output_matrix.shape[1]:
                max_copy_width = transform_data.shape[1]
                if output_matrix.shape[1] < max_copy_width:
                    max_copy_width = output_matrix.shape[1]
                mask_bool = np.not_equal(output_matrix[mask_min_height:(max_copy_height + mask_min_height), mask_min_width:(max_copy_width + mask_min_width)], transform_data)
                masked_matrix = np.ma.MaskedArray(data=output_matrix[mask_min_height:(max_copy_height + mask_min_height), mask_min_width:(max_copy_width + mask_min_width)], mask=mask_bool, fill_value=0)
                output_matrix[mask_min_height:(max_copy_height + mask_min_height), mask_min_width:(max_copy_width + mask_min_width)] = masked_matrix.filled(0)
                mask_min_width = mask_min_width + transform_data.shape[1]
            mask_min_height = mask_min_height + transform_data.shape[0]
        return output_matrix
    # return input matrix if invalid transform
    return input_matrix

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

# returns list of features extracted via dfs in a matrix.
# also needs to be able to find features by splitting matrix into sections
def find_features_in_matrix(input_matrix):
    pass

# in cases where the set of transforms to solve is dependent on some branching feature,
# this function inputs a matrix and branch set and outputs which element in the set
# which the matrix falls under. If it falls under no element in the set, then a
# new element is created in the branch set and returned.
def run_branch_on_matrix(input_matrix, branch_type):
    pass

# create a zeroed out numpy array of width, height dimensions
# numpy arrays are accessed as array[row, column] for reference!!
def create_matrix_of_dimensions(width, height):
    return 

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
test_input = np.array((test_example)["input"])
test_output = np.array((test_example)["output"])

# okay, now the only thing to do is to convert the inputs into an output using a single method...

# Here is an example sequence of transforms to solve 007bbfb7.json
for k in range(training_len + 1):
    input_matrix = test_input
    output_check = test_output
    if(k < training_len):
        input_matrix = training_input[k]
        output_check = training_output[k]

    test_mask = run_transform_on_matrix(input_matrix, transform_type.identity, None)
    output_matrix = run_transform_on_matrix(test_mask, transform_type.scale_board, [9,9])
    output_matrix = run_transform_on_matrix(output_matrix, transform_type.scale, 3)
    output_matrix = run_transform_on_matrix(output_matrix, transform_type.mask, test_mask)

    # output whether output matrix is equivalent to the expected output.
    print(f"The output matrix and test output are equal: {np.array_equiv(output_matrix, output_check)}")
