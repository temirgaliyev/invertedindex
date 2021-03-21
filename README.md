# Inverted Index  
HackNU's Google task: Local search engine

## Dataset  
[Kaggle: All the news](https://www.kaggle.com/snapcrack/all-the-news)

## Basic Usage  
### Precalculate InvertedIndex
```
from index import InvertedIndex
import os
import pickle
import pandas as pd

index_filename = os.path.join('data','inverted_index.pkl')
archive_folder = os.path.join('data', 'archive')
articles_file = os.path.join(archive_folder, 'articles%d.csv')
articles1_df = pd.read_csv(articles_file % 1, usecols=['content'])
articles2_df = pd.read_csv(articles_file % 2, usecols=['content'])
articles3_df = pd.read_csv(articles_file % 3, usecols=['content'])

content = pd.concat([articles1_df['content'],
                     articles2_df['content'],
                     articles3_df['content']]).tolist()
del articles1_df
del articles2_df
del articles3_df

index = InvertedIndex(content)

with open(index_filename, 'wb') as out:
    pickle.dump(index, out, pickle.HIGHEST_PROTOCOL)
```

### Precalculate InvertedIndex
```
import os

query = "presidential elections"
index_filename = os.path.join('data','inverted_index.pkl')

with open(index_filename, 'rb') as inp:
    index = pickle.load(inp)
index.get_relevant_articles('presidential elections')
```
