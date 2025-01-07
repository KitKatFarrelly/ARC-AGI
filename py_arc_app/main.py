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

class feature_type(Enum):
    dfs_feature = 1
    combined_dfs_feature = 2
    matrix_feature = 3

class dfs_feature_data:
    def __init__(self, input_mask, start_height, start_width):
        self.mask = input_mask
        self.highest_point = start_height
        self.lowest_point = start_height
        self.left_point = start_width
        self.right_point = start_width

# dfs on matrix. returns modified visited_matrix
def run_dfs_on_location(input_matrix, visited_mask, start_height, start_width, feature_color):
    if(start_height < 0 or start_width < 0 or start_height >= input_matrix.shape[0] or start_height >= input_matrix.shape[0]):
        return visited_mask
    if(visited_mask.mask[start_height, start_width] == 1):
        return visited_mask
    new_color = input_matrix[start_height, start_width]
    if(feature_color is not new_color):
        return visited_mask
    # add location to mask
    visited_mask.mask[start_height, start_width] = 1
    if(visited_mask.highest_point > start_height):
        visited_mask.highest_point = start_height
    if(visited_mask.lowest_point < start_height):
        visited_mask.lowest_point = start_height
    if(visited_mask.left_point > start_width):
        visited_mask.left_point = start_width
    if(visited_mask.right_point < start_width):
        visited_mask.right_point = start_width
    # run dfs in all 8 directions
    # priorities:
    # 1 2 3
    # 4 _ 5
    # 6 7 8
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height - 1, start_width - 1, feature_color)
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height - 1, start_width, feature_color)
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height - 1, start_width + 1, feature_color)

    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height, start_width - 1, feature_color)
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height, start_width + 1, feature_color)

    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height + 1, start_width - 1, feature_color)
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height + 1, start_width, feature_color)
    visited_matrix = run_dfs_on_location(input_matrix, visited_matrix, start_height + 1, start_width + 1, feature_color)
    return visited_matrix
    

# returns list of features extracted via dfs in a matrix.
# also needs to be able to find features by splitting matrix into sections
def get_feature_list_from_matrix(input_matrix, type, feature_data):
    feature_list = []
    if(type is feature_type.matrix_feature):
        input_matrix_height = input_matrix.shape[0]
        input_matrix_width = input_matrix.shape[1]
        feature_matrix_height = feature_data[0]
        feature_matrix_width = feature_data[1]
        current_height = 0
        while(current_height < input_matrix_height):
            current_width = 0
            while(current_width < input_matrix_width):
                feature_height_end = current_height + feature_matrix_height
                feature_width_end = current_width + feature_matrix_width
                output_matrix = input_matrix[current_height:feature_height_end, current_width:feature_width_end]
                output_location = (output_matrix, current_height, current_width)
                feature_list.append(output_location)
                current_width = current_width + feature_matrix_width
            current_height = current_height + feature_matrix_height
    elif(type is feature_type.dfs_feature):
        # this is just a standard dfs algorithm. edges are determined by color change. adjacency is in all 8 directions.
        input_matrix_height = input_matrix.shape[0]
        input_matrix_width = input_matrix.shape[1]
        # keep track of visited nodes with a matrix of equal size. only used inside the function
        visited_matrix = np.zeros(shape=(input_matrix_height, input_matrix_width))
        current_height = 0
        while(current_height < input_matrix_height):
            current_width = 0
            while(current_width < input_matrix_width):
                if(visited_matrix[current_height, current_width] == 1):
                    continue
                # this should not happen.
                if(visited_matrix[current_height, current_width] > 1):
                    print("invalid value in visited matrix!!!")
                    exit()
                feature_color = input_matrix[current_height, current_width]
                prev_matrix = np.copy(input_matrix)
                feature_mask = dfs_feature_data(visited_matrix, current_height, current_width)
                visited_mask = run_dfs_on_location(input_matrix, feature_mask, current_height, current_width, feature_color)
                feature_mask = np.subtract(visited_mask.mask - prev_matrix)
                # need to crop the mask to the height and width of the feature
                cropped_mask = feature_mask[visited_mask.highest_point:visited_mask.lowest_point, visited_mask.left_point:visited_mask.right_point]
                # re-add color data
                cropped_mask = cropped_mask * input_matrix[current_height, current_width]
                # for now, just output visited mask and location of initial tile.
                output_feature = (cropped_mask, current_height, current_width)
                visited_matrix = visited_mask.mask
                feature_list.append(output_feature)
                current_width = current_width + 1
            current_height = current_height + 1
    return feature_list

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

# Here is an example sequence of transforms to solve 007bbfb7.json
def solve_example_007bbfb7(input_matrix):
    test_mask = run_transform_on_matrix(input_matrix, transform_type.identity, None)
    output_matrix = run_transform_on_matrix(test_mask, transform_type.scale_board, [9,9])
    output_matrix = run_transform_on_matrix(output_matrix, transform_type.scale, 3)
    output_matrix = run_transform_on_matrix(output_matrix, transform_type.mask, test_mask)
    return output_matrix

# This function takes a feature and tries to find it in an input matrix.
# if it does, output qualities of feature in matrix (color, location, shape)
# the important thing is to be able to distinguish the relevant qualities of a feature
# for finding it in a test matrix.
def find_feature_in_matrix(input_matrix, feature_matrix):
    # find size of feature. this gives us a patch size to check against on the input matrix
    feature_height = feature_matrix.shape[0]
    feature_width = feature_matrix.shape[1]

    # return early if feature matrix is larger than input matrix
    if(feature_matrix.shape[0] > feature_height or feature_matrix.shape[1] > feature_width):
        return []

    # loop through each feature-sized patch in the input matrix.
    feature_start_height = 0
    while(feature_start_height <= (input_matrix.shape[0] - feature_height)):
        feature_start_width = 0
        while(feature_start_width <= (input_matrix.shape[1] - feature_width)):
            feature_stop_height = feature_start_height + feature_height
            feature_stop_width = feature_start_width + feature_width
            check_matrix = input_matrix[feature_start_height:feature_stop_height, feature_start_width:feature_stop_width]
            # obviously an exact match is not particularly likely to be found. This needs to be made more robust
            if(np.array_equiv(check_matrix, feature_matrix)):
                # maybe replace this with a yield, or make a list of matches. In case of multiple matches.
                # for now, just state coordinates where the feature was found.
                return [feature_start_height, feature_start_width]
            feature_start_width = feature_start_width + 1
        feature_start_height = feature_start_height + 1
    return []


class trait_type(Enum):
    symmetry = 1
    color = 2
    similarity = 3
    rotate = 4
    mirror = 5
    size = 6
    translational = 7
    occlusion = 8

class matrix_trait_type(Enum):
    size = 1
    feature_count = 2
    copies_of_feature = 3

def find_feature_from_traits(input_matrix, feature_traits):
    for traits in feature_traits:
        if(traits[0] is trait_type.similarity):
            return input_matrix[traits[1]:(traits[1] + traits[3]), traits[2]:(traits[2] + traits[4])]
    return []
            

def convert_traits_to_coordinates(feature_traits):
    pass

# this function just decides the correct actions to take given a list of features, and a list of traits of those features
# this function will decide whether to do rotation, cropping, translation, etc depending on similar and differing traits
# outputs the action to take and the relevant transform data
def decide_action_for_features(feature_list, feature_traits, matrix_traits):
    pass

class trait_object:
    def __init__(self, trait, trait_data):
        self.trait = trait
        self.trait_data = trait_data

# this function just tries to find the similar traits between a set of features
# ideally, for these comparison functions, the program should be able to discover
# different types of traits between features and matrices.
# maybe attention matrices could be used for this task?
# feature traits should be able to be simplified and transformationally disctinct from other traits.
# feature traits should be defined robustly in terms of attention - they should start overtuned and
# become more general as they are trained.
def compare_feature_traits(list_of_features_to_compare):
    # traits to compare:
    # symmetry
    # color
    # similarity
    # rotational/mirrored
    # size
    # subfeatures inside feature
    # relative/absolute location
    # occlusion vs another feature

    # first find size of feature list and create a 2D matrix of feature trait lists
    number_of_features = len(list_of_features_to_compare)
    output_traits = [[[] for i in range(number_of_features)] for j in range(number_of_features)]

    # next, iterate over every combination of 2 features and create an object of their shared traits
    for current_row in range(number_of_features):
        for current_column in range(number_of_features):
            # prevents duplicate trait finding
            if((current_row > current_column) or (current_row == current_column)):
                continue
            first_feature = list_of_features_to_compare[current_row]
            second_feature = list_of_features_to_compare[current_column]
            if(np.array_equiv(first_feature[0], second_feature[0])):
                (output_traits[current_row][current_column]).append((trait_type.similarity, first_feature[1], first_feature[2], first_feature[0].shape[0], first_feature[0].shape[1]))
            if(first_feature[3] is feature_type.dfs_feature):
                # detect if the masks are the same, if colors are the same, etc
                pass

    return output_traits

# this function just tries to find the similar traits between a set of matrices
def compare_matrix_traits(list_of_matrices_to_compare):
    # traits to compare:
    # size
    # number of features
    # number of copies of any given feature
    pass

def open_test_file_and_test(input_dir, input_file):

    input_path = os.path.join(input_dir, input_file)
    with open(input_path, 'r') as file:
        data = json.load(file)

    print(f"opened {input_file}...")

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

    # parse input and output arrays per example
    training_input = {}
    training_output = {}
    for j in range(training_len):
        training_input[j] = np.array((training_example_list[j])["input"])
        training_output[j] = np.array((training_example_list[j])["output"])
    test_input = np.array((test_example)["input"])
    test_output = np.array((test_example)["output"])

    # okay, now the only thing to do is to convert the inputs into an output using a single method...
    known_traits = []

    # iterate over all examples in a training set then try to solve the test example.
    for k in range(training_len + 1):
        input_matrix = test_input
        output_check = test_output
        if(k < training_len):
            input_matrix = training_input[k]
            output_check = training_output[k]
        
        found_feature = find_feature_from_traits(input_matrix, known_traits)

        was_feature_found = (np.array_equiv(found_feature, output_check))

        if(k < training_len):
            # need to find all features here
            # TODO: add dfs features
            # NOTE: for now, just get list of features based on output feature
            feature_list = get_feature_list_from_matrix(input_matrix, feature_type.matrix_feature, output_check.shape)

            # TODO: for now, just append output check to feature list
            feature_list.append((output_check, 0, 0))

            trait_matrix = compare_feature_traits(feature_list)

            new_known_traits = []

            object_column = len(feature_list)
            for current_row in range(object_column):
                if(len(trait_matrix[current_row][object_column - 1]) > 0):
                    print(f"trait list at location: {trait_matrix[current_row][object_column - 1]}")
                    new_known_traits.append((trait_matrix[current_row][object_column - 1])[0])

            # create set of known traits
            if(k == 0):
                known_traits = new_known_traits
            else:
                known_traits = set(known_traits) & set(new_known_traits)

            if not known_traits:
                print(f"no matching traits found between input features and output for {input_file}")
                
                break
        else:
            # the final training set should just test if we found the solution from known features:
            if was_feature_found:
                # output whether output matrix is equivalent to the expected output.
                print(f"feature was found with trait: {known_traits}")
                return True
            else:
                return False

        #compare traits of correct feature with output
        #compare_feature_traits([output_check, correct_feature])

        #compare traits of correct feature in input with incorrect features in input
        #compare_feature_traits(feature_list)

parser = argparse.ArgumentParser()
parser.add_argument("inputdir", help="input path to the test directory")

args = parser.parse_args()
input_dir = os.path.abspath(args.inputdir)

solved_problems = 0

if(os.path.isdir(input_dir)):
    for input_file in os.listdir(input_dir):
        if open_test_file_and_test(input_dir, input_file):
            solved_problems = solved_problems + 1
        
elif(os.path.isfile(input_dir)):
    head_tail = os.path.split(input_dir)
    if open_test_file_and_test(head_tail[0], head_tail[1]):
        solved_problems = solved_problems + 1
else:
    print("not a directory or file!")

print(f"number of solved problems is now {solved_problems}")
