#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#                                                                             
# PROGRAMMER: Mateusz Rogalski
# DATE CREATED: 08.02.2020
# REVISED DATE: 09.02.2020
# PURPOSE: Create a function calculates_results_stats that calculates the 
#          statistics of the results of the programrun using the classifier's model 
#          architecture to classify the images. This function will use the 
#          results in the results dictionary to calculate these statistics. 
#          This function will then put the results statistics in a dictionary
#          (results_stats_dic) that's created and returned by this function.
#          This will allow the user of the program to determine the 'best' 
#          model for classifying the images. The statistics that are calculated
#          will be counts and percentages. Please see "Intro to Python - Project
#          classifying Images - xx Calculating Results" for details on the 
#          how to calculate the counts and percentages for this function.    
#         This function inputs:
#            -The results dictionary as results_dic within calculates_results_stats 
#             function and results for the function call within main.
#         This function creates and returns the Results Statistics Dictionary -
#          results_stats_dic. This dictionary contains the results statistics 
#          (either a percentage or a count) where the key is the statistic's 
#           name (starting with 'pct' for percentage or 'n' for count) and value 
#          is the statistic's value.  This dictionary should contain the 
#          following keys:
#            n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds
#            pct_match - percentage of correct matches
#            pct_correct_dogs - percentage of correctly classified dogs
#            pct_correct_breed - percentage of correctly classified dog breeds
#            pct_correct_notdogs - percentage of correctly classified NON-dogs
#
##
def do_labels_match(result_value):
    return result_value[2] == 1


def is_really_dog(result_value):
    return result_value[3] == 1


def is_classified_dog(result_value):
    return result_value[4] == 1


def filter_and_get_count(collection, condition_func):
    return sum([1 for item in collection if condition_func(item)])


def get_real_dog_count(result_dict):
    return filter_and_get_count(result_dict.values(), is_really_dog)


def get_full_match_count(result_dict):
    return filter_and_get_count(result_dict.values(), do_labels_match)


def get_isdog_match_count(result_dict):
    is_isdog_match = lambda result_value: is_really_dog(result_value) and is_classified_dog(result_value)
    return filter_and_get_count(result_dict.values(), is_isdog_match)


def get_notdog_match_count(result_dict):
    is_notdog_match = lambda result_value: not is_really_dog(result_value) and not is_classified_dog(result_value)
    return filter_and_get_count(result_dict.values(), is_notdog_match)


def get_breed_match_count(result_dict):
    is_breed_match = lambda result_value: do_labels_match(result_value) and is_classified_dog(result_value)
    return filter_and_get_count(result_dict.values(), is_breed_match)


def calculate_percentage(part, of_total):
    if part == 0:
        return 0
    else:
        return part/of_total * 100


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using classifier's model 
    architecture to classifying pet images. Then puts the results statistics in a 
    dictionary (results_stats_dic) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value. See comments above
                     and the previous topic Calculating Results in the class for details
                     on how to calculate the counts and statistics.
    """        
    # Replace None with the results_stats_dic dictionary that you created with 
    # this function
    stats = {}
    stats["n_images"] = len(results_dic)
    stats["n_dogs_img"] = get_real_dog_count(results_dic)
    stats["n_notdogs_img"] = stats["n_images"] - stats["n_dogs_img"]
    stats["n_match"] = get_full_match_count(results_dic)
    stats["n_correct_dogs"] = get_isdog_match_count(results_dic)
    stats["n_correct_notdogs"] = get_notdog_match_count(results_dic)
    stats["n_correct_breed"] = get_breed_match_count(results_dic)
    stats["pct_match"] = calculate_percentage(stats["n_match"], stats["n_images"])
    stats["pct_correct_dogs"] = calculate_percentage(stats["n_correct_dogs"], stats["n_dogs_img"])
    stats["pct_correct_breed"] = calculate_percentage(stats["n_correct_breed"], stats["n_dogs_img"])
    stats["pct_correct_notdogs"] = calculate_percentage(stats["n_correct_notdogs"], stats["n_notdogs_img"])
    return stats
