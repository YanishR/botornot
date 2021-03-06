# botornot
Final project for CMSC 395: Natural Language Processing

This program intends to determine whether a tweet
is generated by a bot or not

Dependencies:
--------------------
for Deep Learning:
--------------------
1. gensim / word2vec
2. tensorflow
3. keras
4. numpy
5. tqdm


Run:
--------------------
for Deep Learning:
- construct RNN() with real tweet file and troll tweet file.
- run() with deafult settings.

default parameters:
- embedding_dim = 32 # embedding dimension
- max_length = 200 # tweet max length 
- trunc_type = 'post' # truncate at the end
- padding_type = 'post' # padding at the end
- oov_tok = '<OOV>' # OOV = Out of Vocabulary
- vocab_size = 5000 # top common words
- epochs = 3
- batch_size = 16
- hidden neurons = 32
- no droup out layer

note: all hyperparameters and architecture can be changed in RNN.py
--------------------

----------------------------------------
SVM Classification:
----------------------------------------
Dependencies:
    - sklearn
    - numpy

----------------------------------------
Running SVM classification
For SVM classification:
 - Create Vectorizer with real tweet file and troll tweet file
 - sizes for both are optional. Default is 5000 and 1000 respectively
 - And then run either runSVM() or runKBestFeatures()
 - Example is given in botornot.py and at the bottom of Vectorizer.py
    
runSVM inputs in that order. All have defaults:
    - contentMatrix,
    - contentMatrixFeatures=[],
    - wordN=1,
    - stylisticMatrix=True, 
    - stylisticMatrixFeatures=[], 
    - charN=3, 
    - testSize=.3,
    - r=1, 
    - hyper_param=0.1, 
    - kernel_type=0, kernel_type=0 is linear, 1 is poly
    - degree=1
For contentMatrixFeatures and stylisticMatrixFeatures,
    the list should be of integers only.

contenMatrixFeatures feature correspondence:
 - 0: average number of emjojis
 - 1: average number of URLs
 - 2: number of hastags
 - 3: POS tag distribution
 - 4: Number of tokens

stylisticMatrixFeatures feature correspondence
 - 0: average number of punctuation 
 - 1: average word size
 - 2: vocab size 
 - 3: POS tag distribution
 - 4: digit frequency
 - 5: average hashtag length
 - 6: average capitalizations
 - 7: letter frequency

Modules:
    Data from dataparser.py:
        - Reads and manages data usage
