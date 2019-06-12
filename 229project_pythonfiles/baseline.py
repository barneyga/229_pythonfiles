from __future__ import division, print_function
import argparse
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random
import pandas as pd
import pickle


NUM_TRIALS = 100

def main():
    acc_sum = 0
    # process data into matrix with shit, for now just a randomly intitialized matrix so it compiles

    pickle_file = open('labels.pkl', 'rb')
    genre_dict = pickle.load(pickle_file)
    data = []
    genres = set()
    for key in genre_dict.keys():
        l = []
        l.append(key)
        temp = []
        for item in genre_dict[key]:
            genres.add(item)
            temp.append(item)
        l.append(temp)
        data.append(l)
    print(data[0])
    # make it into list so that random.choice() works.
    genres = list(genres)
    # genres = set()
    # # change when we get the data processing function
    # # assuming data processing function returns matrix with ind 0 as title and ind 1 as genres
    # # also assuming that we have genre names w/out the id's 
    # for i in range(len(data)):
    #     for x in data[i]:
    #         genres.add(data[i][x])
    # tests baseline function
    random.seed(229)
    for i in range(NUM_TRIALS):
        acc = 0
        for n in range(len(data)):
            pred = random.choice(genres)
            # i should be the corresponding index for the genre
            # ideally, preprocessing function makes it so title is at 0 and genre is at 1
            # i = None
            # data[i][1][0] is the first word in the list of genres associated with a movie i
            if pred in data[n][1]:
                acc +=1
        acc = acc/len(data)
        print("Accuracy for trial #" + str(i) + ":" + str(acc))
        acc_sum += acc
    print("Total accuracy:" + str(acc_sum/NUM_TRIALS))
    return 

main()