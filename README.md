# Search Engine using Inverted Index and TF-IDF  
HackNU's Google task: Local search engine

## Dataset  
[Kaggle: All the news](https://www.kaggle.com/snapcrack/all-the-news)  
Note: that's only sample dataset, SearchEngine class may work with any iterable of strings

## Basic Usage  
### Precalculate Inverted Index and TF-IDF
```
from search import SearchEngine
import os
import pickle
import pandas as pd

engine_filename = os.path.join('data','search_engine.pkl')
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

engine = SearchEngine(content)  # might take long time to create Inverted Index and TF-IDF vectors

with open(engine_filename, 'wb') as out:
    pickle.dump(engine, out, pickle.HIGHEST_PROTOCOL)
```

### Use SearchEngine
```
import os
import pickle

query = 'presidential elections'
engine_filename = os.path.join('data', 'search_engine.pkl')

with open(engine_filename, 'rb') as inp:
    engine = pickle.load(inp)
engine.get_relevant_articles(query)  # returns list of 5 tuples (cosine_score, document_id)
```
