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

# turns images into a SIZExSIZE image
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

    #     # requests the image from its url found in the csv and saves it to our images/ directory.
    #     # url = 'https://image.tmdb.org/t/p/w500' + url
    #     # response = requests.get(url)
    #     # if response.status_code == 404 or response.status_code == 403:
    #     #     print('added ' + title + ' to the blacklist')
    #     #     blacklist.append(title)
    #     #     continue
    #     # img = Image.open(BytesIO(response.content))
    #     # if img.format == 'PNG':
    #     #     print('.png file found at ' + str(i) + '-' + title)
    #     #     img = img.convert('RGB')
    #     # img.save('images/' + str(i) + '-' + title + '.jpg')
    #     if (i+1) % 10 == 0: 
    #         print(i+1)

    # print(blacklist)
    # print(genre_set)
    # print(len(genre_set))
    # print(len(genre_dict))
    # # https://pythonspot.com/save-a-dictionary-to-a-file/
    # f = open("labels.pkl","wb")
    # pickle.dump(genre_dict,f)
    # f.close()
    # toCSV = []
    # i = 0
    # for title in genre_dict.keys():
    #     d = {}
    #     d['Movie'] = str(i)+'-'+title
    #     image = Image.open('images224x224/'+d['Movie']+'.jpg')
    #     for genre in genre_set:
    #         output_dir = './data/genre_data/' + genre
    #         if genre in genre_dict[title]:
    #             image.save(os.path.join(output_dir, d['Movie'] + '.jpg'))
    #             # d[genre] = 1
    #         # else:
    #         #     d[genre] = 0
    #     i += 1
    #     # toCSV.append(d)
    # keys = toCSV[0].keys()
    # with open('milestone_labels.csv', 'w') as output_file:
    #     dict_writer = csv.DictWriter(output_file, keys, lineterminator = '\n')
    #     dict_writer.writeheader()
    #     dict_writer.writerows(toCSV)

        



    # TODO: THIS IS DISGUSTING dont look at it
    # l = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science_Fiction', 'Thriller', 'TV_Movie', 'War', 'Western']
    # for cur in l:
    #     train = []
    #     print(cur)
    #     f = open("movies_by_genre/" + cur + ".txt", "r", encoding='utf-16')
    #     for line in tqdm(f):
    #         line = line.strip()
    #         line = line.replace('.jpg', '')
    #         title = line.split('-', 1)[-1]
    #         genres = genre_dict[title]
    #         l = []
    #         if 'War' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Fantasy' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Mystery' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'TV Movie' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Science Fiction' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Western' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Comedy' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Documentary' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Crime' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Action' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Music' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Adventure' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Family' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Thriller' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'History' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Horror' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Foreign' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Drama' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Romance' in genres:
    #             l.append('1,')
    #         else:
    #             l.append('0,')
    #         if 'Animation' in genres:
    #             l.append('1')
    #         else:
    #             l.append('0')
    #         s = ''.join(l) + '\n'
    #         s = s.replace(',', ' ')
    #         train.append(s)
    #     f = open('movies_by_genre/' + cur + '_Labels.txt', 'w')
    #     f.writelines(train)
    #     f.close()

    # filenames = os.listdir('images224x224/')
    # dirnames = os.listdir('data/genre_data/')
    # random.seed(229)
    # train_overall, dev_overall, test_overall = set(), set(), set()
    # for dirname in tqdm(dirnames):
    #     filenames = os.listdir('data/genre_data/' + dirname + '/')
    #     filenames.sort()
    #     random.shuffle(filenames)  # randomly shuffles the ordering of filenames

    #     split_1 = int(0.8 * len(filenames))
    #     split_2 = int(0.9 * len(filenames))
    #     train_filenames = filenames[:split_1]
    #     dev_filenames = filenames[split_1:split_2]
    #     test_filenames = filenames[split_2:]
    #     for filename in tqdm(train_filenames):
    #         train_overall.add(filename + '\n')
    #         # resize_and_save(filename, 'data/sets/train/' + dirname, size=64)
    #     for filename in tqdm(dev_filenames):
    #         dev_overall.add(filename + '\n')
    #         # resize_and_save(filename, 'data/sets/dev/' + dirname, size=64)
    #     for filename in tqdm(test_filenames):
    #         test_overall.add(filename + '\n')
    #         # resize_and_save(filename, 'data/sets/test/' + dirname, size=64)
    # train_overall, dev_overall, test_overall = list(train_overall), list(dev_overall), list(test_overall)
    # f = open('train_files.txt', 'w')
    # f.writelines(train_overall)
    # f.close()
    # f = open('dev_files.txt', 'w')
    # f.writelines(dev_overall)
    # f.close()
    # f = open('test_files.txt', 'w')
    # f.writelines(test_overall)
    # f.close()

    # train, dev, test = [], [], []
    # f = open("train_files.txt", "r", encoding='utf-16')
    # for line in tqdm(f):
    #     line = line.strip()
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
    #     train.append(s)
    # f = open('train_labels.txt', 'w')
    # f.writelines(train)


    # f = open("dev_files.txt", "r", encoding='utf-16')
    # for line in tqdm(f):
    #     line = line.strip()
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
    #     dev.append(s)
    # f = open('dev_labels.txt', 'w')
    # f.writelines(dev)

    # f = open("test_files.txt", "r", encoding='utf-16')
    # for line in tqdm(f):
    #     line = line.strip()
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
    #     test.append(s)
    # f = open('test_labels.txt', 'w')
    # f.writelines(test)

    # split_1 = int(0.8 * len(filenames))
    # split_2 = int(0.9 * len(filenames))
    # train_filenames = filenames[:split_1]
    # dev_filenames = filenames[split_1:split_2]
    # test_filenames = filenames[split_2:]

    # taken from cs230 project
    # Preprocess train, dev and test
    # for filename in tqdm(train_filenames):
    #     resize_and_save(filename, 'train224x224/', size=64)
    # for filename in tqdm(dev_filenames):
    #     resize_and_save(filename, 'dev224x224/', size=64)
    # for filename in tqdm(test_filenames):
    #     resize_and_save(filename, 'test224x224/', size=64)

    # for filename in tqdm(filenames):
    #     resize_and_save(filename, 'images224x224/', size=64)