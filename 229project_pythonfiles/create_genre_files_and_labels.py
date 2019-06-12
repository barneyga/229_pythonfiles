from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import collections
import ast
import os
import random
from tqdm import tqdm
import pickle
import csv
import numpy as np
import collections

def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open('images224x224/'+filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    # image = image.resize((224, 224), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

if __name__ == '__main__':
    df = pd.read_csv('the-movies-dataset/movies_metadata.csv', encoding = "ISO-8859-1", usecols=['genres', 'original_title', 'poster_path'])
    desired_size = float('inf')
    blacklist = [] # this blacklist is for images that 404
    genre_dict = collections.defaultdict(list)

    genre_set = set()

    for i in range(len(df)):
        # Dictionary formatting in csv: [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]
        genres = str(df['genres'][i])
        url = str(df['poster_path'][i])
        title = str(df['original_title'][i])

        # parses into a list of dictionaries of a movie's genres and then stores them in our better formatted dictionary.
        genres = ast.literal_eval(genres)
        for d in genres:
            genre_set.add(d['name'])
            genre_dict[title].append(d['name'])

    genre_train_dict = collections.defaultdict(list)
    d = collections.defaultdict(list)
    f = open("train_files.txt", "r", encoding='utf-16')
    for line in tqdm(f):
        line = line.strip()
        og = line
        line = line.replace('.jpg', '')
        title = line.split('-', 1)[-1]
        genres = genre_dict[title]
        l = []
        if 'War' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Fantasy' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Mystery' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'TV Movie' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Science Fiction' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Western' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Comedy' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Documentary' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Crime' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Action' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Music' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Adventure' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Family' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Thriller' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'History' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Horror' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Foreign' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Drama' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Romance' in genres:
            l.append('1,')
        else:
            l.append('0,')
        if 'Animation' in genres:
            l.append('1')
        else:
            l.append('0')
        s = ''.join(l) + '\n'
        s = s.replace(',', ' ')
        if 'Comedy' in genres or 'Drama' in genres:
            continue
            
        for g in genres:
            genre_train_dict[g].append(s)
            d[g].append(og+'\n')
    for genre, title_list in tqdm(d.items()):
        f = open('training_movies_by_genre/'+ genre + '_noDrama_noComedy_files.txt', 'w')
        f.writelines(title_list)
    for genre, label_list in tqdm(genre_train_dict.items()):
        f = open('training_movies_by_genre/'+ genre + '_noDrama_noComedy_labels.txt', 'w')
        f.writelines(label_list)





























    # genre_train_dict = collections.defaultdict(list)
    # d = collections.defaultdict(list)
    # f = open("train_files.txt", "r", encoding='utf-16')
    # for line in tqdm(f):
    #     line = line.strip()
    #     og = line
    #     line = line.replace('.jpg', '')
    #     title = line.split('-', 1)[-1]
    #     genres = genre_dict[title]
    #     l = []
    #     if 'War' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Fantasy' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Mystery' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'TV Movie' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Science Fiction' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Western' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Comedy' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Documentary' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Crime' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Action' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Music' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Adventure' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Family' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Thriller' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'History' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Horror' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Foreign' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Drama' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Romance' in genres:
    #         l.append('1,')
    #     else:
    #         l.append('0,')
    #     if 'Animation' in genres:
    #         l.append('1')
    #     else:
    #         l.append('0')
    #     s = ''.join(l) + '\n'
    #     s = s.replace(',', ' ')
    #     for g in genres:
    #         genre_train_dict[g].append(s)
    #         d[g].append(og+'\n')
    # for genre, title_list in tqdm(d.items()):
    #     f = open('training_movies_by_genre/'+ genre + '_files.txt', 'w')
    #     f.writelines(title_list)
    # for genre, label_list in tqdm(genre_train_dict.items()):
    #     f = open('training_movies_by_genre/'+ genre + '_labels.txt', 'w')
    #     f.writelines(label_list)