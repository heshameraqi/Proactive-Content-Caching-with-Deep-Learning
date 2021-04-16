import re
import time
import random
from collections import Counter
from itertools import repeat, chain


# we recreate new ids for movies starting from 1 to avoid gap in movies ids in the original file
def data_preparation_movie_rating(file_name1, file_name2):
    file_trace = open(file_name1, "r")
    user_id_list = list()
    movie_id_list = list()
    contents = file_trace.readlines()
    for x in contents:
        user_id = int(x.split('::')[0])
        movie_id = int(x.split('::')[1])
        user_id_list.append(user_id)
        movie_id_list.append(movie_id)
    file_trace.close()
    movie_id_list_unique = list(dict.fromkeys(movie_id_list))
    file_trace_formatted = open(file_name2, "w")
    for i in range(0, len(user_id_list)):
        new_movie_id = movie_id_list_unique.index(movie_id_list[i]) + 1  # we add one to start identifiying elements from 1 and not 0
        line_content = str(user_id_list[i]) + '\t' + str(new_movie_id) + '\n'
        file_trace_formatted.write(line_content)
    file_trace_formatted.close()


# we transform user profile data into more suitable format for the DL model
def data_preparation_user_profile(file_name1, file_name2):
    file_trace = open(file_name1, "r")
    user_id_list = list()
    user_gender_list = list()
    user_age_list = list()
    user_occupation_list = list()
    user_zip_code_original_list = list()
    contents = file_trace.readlines()
    for x in contents:
        user_id = int(x.split('::')[0])
        user_gender = x.split('::')[1]
        user_age = int(x.split('::')[2])
        user_occupation = int(x.split('::')[3])
        # user_zip_code = int(x.split('::')[4])
        user_zip_code = int(x.split('::')[4][0:5])
        user_id_list.append(user_id)
        if user_gender == 'M':
            user_gender_list.append(1)
        else:
            user_gender_list.append(2)
        user_age_list.append(age_switcher(user_age))
        user_occupation_list.append(user_occupation)
        user_zip_code_original_list.append(user_zip_code)
    file_trace.close()
    user_zip_code_original_unique = list(dict.fromkeys(user_zip_code_original_list))  # to keep only one copy of each requested item
    user_zip_code_list = list()
    for i in range(0, len(user_zip_code_original_list)):
        new_value = user_zip_code_original_unique.index(user_zip_code_original_list[i]) + 1  # we add one to start identifiying elements from 1 and not 0
        user_zip_code_list.append(new_value)
    file_trace_formatted = open(file_name2, "w")
    for i in range(0, len(user_id_list)):
        # val = str(seq_req[i]) + '\n'
        val = str(user_id_list[i]) + '\t' + str(user_gender_list[i]) + '\t' + str(user_age_list[i]) + '\t' + str(user_occupation_list[i]) + '\t' + str(user_zip_code_list[i]) + '\n'
        file_trace_formatted.write(val)
    file_trace_formatted.close()


def age_switcher(i):
    switcher = {
        1: 1,
        18: 2,
        25: 3,
        35: 4,
        45: 5,
        50: 6,
        56: 7
    }
    return switcher.get(i, "Invalid age")


if __name__ == "__main__":
    # data_preparation("test.txt", "test_result.txt")
    # data_preparation_user_profile("Users_profiles_original.txt", "Users_profiles.txt")
    data_preparation_movie_rating("MovieLens_rating_original.txt", "MovieLens_rating.txt")