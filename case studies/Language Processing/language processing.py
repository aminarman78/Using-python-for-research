# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:32:34 2020

@author: Amin
"""
text = "Hi my name is Mohammad Amin and I'm a chemical engineering student  student student"
def count_words(text):
    """"count the number of times each word occurs in text and Return a 
    dictionary where the keys are uniqe words and values are word counts.Skip punctuations"""
    text = text.lower()
    skips = ['.', ',', ';', ':', '"', "'"]
    for ch in skips:
        text = text.replace(ch, '')
        
    word_count = {}
    for word in text.split(' '):
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

from collections import Counter

def count_words_fast(text):
    text = text.lower()
    skips = ['.', ',', ';', ':', '"', "'",'\n']
    for ch in skips:
        text = text.replace(ch, '')
    word_count = Counter(text.split(' '))
    return word_count
# =============================================================================
# 
# =============================================================================

def read_book(title_path):
    """Read a book and return it as a string"""
    with open(title_path, 'r', encoding='utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n', '').replace('\r', '')
    return text

# text = read_book('romeo.txt')
# ind = text.find("What's in a name?")
# sample_text = text[ind : ind +300]
# trump = read_book('trump.txt')



def word_stats(word_counts):
    """Return number of unique words and word frequencies"""
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)

text = read_book('romeo.txt')
word_counts = count_words(text)
(num_unique,counts) = word_stats(word_counts)
# print(num_unique)
# print(sum(counts))

# =============================================================================
# 
# =============================================================================

# ###Reading multiple files
# import pandas as pd
# import os
# book_dir = "Books"
# stats = pd.DataFrame(columns = ('language', 'author', 'title', 'length', 'unique'))
# title_num = 1
# for language in os.listdir(book_dir):
#     for author in os.listdir(book_dir + "/" + language + "/"):
#         for title in os.listdir(book_dir + "/" + language + "/" + author):
#             inputfile = book_dir + "/" + language + "/" + author + "/" + title 
#             text = read_book(inputfile)
#             (num_unique, counts) = word_stats(count_words(text))
#             stats.loc[title_num] = language, author.capitalize(), title.replace('.txt', ''), sum(counts), num_unique
#             title_num += 1
# head = stats.head()
# bottom = stats.tail()
# ###Pandas         
# table = pd.DataFrame(columns=('name','age'))
# table.loc[1] = 'Mohammad Amin', 21
# table.loc[2] = 'Arman', 10
# print(table)

# # =============================================================================
# # 
# # =============================================================================
# ###plotting books statistics

# import matplotlib.pyplot as plt
# plt.plot(stats.length, stats.unique, 'bo')
# plt.loglog(stats.length, stats.unique, 'bo')
# stats[stats.language == 'English']

# plt.figure(figsize = (10, 10))
# subset = stats[stats.language == 'English']
# plt.loglog(subset.length, subset.unique, 'o', label= 'English', color='crimson')
# subset = stats[stats.language == 'French']
# plt.loglog(subset.length, subset.unique, 'o', label= 'French', color='forestgreen')
# subset = stats[stats.language == 'German']
# plt.loglog(subset.length, subset.unique, 'o', label= 'German', color='orange')
# subset = stats[stats.language == 'Portuguese']
# plt.loglog(subset.length, subset.unique, 'o', label= 'Portuguese', color='blueviolet')
# plt.legend()
# plt.xlabel('Number of words')
# plt.ylabel('Number of unique words')
# plt.savefig('bookstats.pdf')

# =============================================================================
# 
# =============================================================================
## Homework
#1
import pandas as pd


hamlets = pd.read_csv('hamlet.csv',index_col=0)
##2
language, text = hamlets.iloc[0]
counted_text = count_words_fast(text)
##3
def word_frequency(counts):
    if (counts > 10):
        frequency="frequent"
    elif(1<counts<=10):
        frequency="infrequent"
    else:
        frequency="unique"
    return(frequency)

data = pd.DataFrame()
data['word'] = counted_text.keys()
data['count'] = counted_text.values()
data['length'] = [len(word) for word in counted_text.keys()]
data['frequency'] = [word_frequency(value) for value in counted_text.values()]

unique = 0
for count in list(counted_text.values()):
    if word_frequency(count) == 'unique':
        unique += 1
##4
sub_data = pd.DataFrame()
sub_data['language'] = ['English' for i in range(3)]
sub_data['frequency'] = ['frequent', 'infrequent', 'unique']
num_frequent = 0
for frequency in list(data['frequency']):
    if frequency == 'frequent':
        num_frequent += 1
num_infrequent = 0  
for frequency in list(data['frequency']):
    if frequency == 'infrequent':
        num_infrequent += 1
num_unique = 0
for frequency in list(data['frequency']):
    if frequency == 'unique':
        num_unique += 1 
frequent = []
infrequent = []
unique = []
for word,count in counted_text.items():
    if word_frequency(count) == 'frequent':
        frequent.append(word)
    elif word_frequency(count) == 'infrequent':
        infrequent.append(word)
    else:
        unique.append(word)
        
        
frequent_length = 0
infrequent_length = 0
unique_length = 0


for word in frequent:
    frequent_length += len(word)
infrequent_length = 0
for word in infrequent:
    infrequent_length += len(word)
for word in unique:
    unique_length += len(word)

mean_frequent = frequent_length / num_frequent
mean_infrequent = infrequent_length / num_infrequent
mean_unique = unique_length / num_unique
sub_data['mean_word_length'] = [mean_frequent, mean_infrequent, mean_unique]
sub_data['num_words'] = [num_frequent, num_infrequent , num_unique]
##5

def summarize_text(language, text):
    counted_text = count_words_fast(text)

    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })
    
    data.loc[data["count"] > 10,  "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1,  "frequency"] = "unique"
    
    data["length"] = data["word"].apply(len)
    
    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ["frequent","infrequent","unique"],
        "mean_word_length": data.groupby(by = "frequency")["length"].mean(),
        "num_words": data.groupby(by = "frequency").size()
    })
    
    return(sub_data)


grouped_data = pd.DataFrame(columns = ["language", "frequency", "mean_word_length", "num_words"])
# I used "grouped_data=pd.DataFrame({})"
for loop in range (hamlets.shape[0]): #and "(0, len(hamlets))" which gave me an identical output, but datacamp didn't like it.
    language, text = hamlets.iloc[loop]
    sub_data = summarize_text(language, text)
    grouped_data = grouped_data.append(sub_data)
# 6: Plot the stats

colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o","infrequent": "s", "unique": "^"}
import matplotlib.pyplot as plt
for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words,
        marker=markers[row.frequency],
        color = colors[row.language],
        markersize = 10
    )

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
        color=colors[color],
        marker="o",
        label = color, markersize = 10, linestyle="None")
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
        color="k",
        marker=markers[marker],
        label = marker, markersize = 10, linestyle="None")
    )
plt.legend(numpoints=1, loc = "upper left")

plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
plt.show()




