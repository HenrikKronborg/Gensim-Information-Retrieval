import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
import gensim

random.seed(123)

# ------------
# Functions
# ------------
# Clean paragraphs and split paragraph into words
def filterInput(input):
    # Convert paragraph to lower case
    input = input.lower()
    # Remove newline
    input = input.replace("\r\n", " ")
    # Remove tab
    input = input.replace("\t", " ")
    # Remove punctuation. Trick where you join with an empty string and only retrieve non-punctuation characters
    input = ''.join(character for character in input if character not in string.punctuation)
    # Split paragraph into an array of words
    input = input.split()
    return input

# Stem the words in given paragraph
def stem(input):
    # For each word, stem it
    for i, word in enumerate(input):
        input[i] = stemmer.stem(word)
    return input

# Print 5 lines of a paragraph from the original paragraphs
def printRelevantParagraphs(docSimilarity):
    for document, weight in docSimilarity:
        paragraph = originalParagraphs[document]
        paragraph = paragraph.split("\r\n")

        print("[ Paragraph " + str(document) + " ]")
        for i in range(0, min(len(paragraph), 5)):
            print(paragraph[i])
        print()

# ------------
# 1. Data loading and preprocessing
# ------------
f = codecs.open("pg3300.txt", "r", "utf-8")

stemmer = PorterStemmer()
paragraphs = []

# Remove header and footers. Split on double newlines
for paragraph in f.read().split("\r\n\r\n"):
    # If the word "gutenberg" is found in paragraph, don't include it in the new array
    if "gutenberg" not in paragraph.lower():
        paragraphs.append(paragraph)

# Keep a copy of the original paragraphs
originalParagraphs = paragraphs.copy()

for i, paragraph in enumerate(paragraphs):
    # Filter the current paragraph, using the function above
    paragraphs[i] = filterInput(paragraphs[i])

    # Create a copy of the current paragraph
    words = paragraphs[i]

    # For each word in the copy, stem it
    for j, word in enumerate(words):
        # Stem the words
        words[j] = stemmer.stem(word)

    # Set the current paragraph to be the copy that contains stemmed words
    paragraphs[i] = words


# ------------
# 2. Dictionary building
# ------------
stoplist = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your".split(",")
bagsOfWords = []

# Build a dictionary on all the paragraphs. The dictionary contains integers representing unique words
dictionary = gensim.corpora.Dictionary(paragraphs)

stopwordIds = []
for stopword in stoplist:
    # If stopword is found in dictionary, add its ID to the stopwordIds array
    if stopword in dictionary.token2id:
        stopwordIds.append(dictionary.token2id[stopword])

# Remove stopwords from the dictionary
dictionary.filter_tokens(stopwordIds)

# Create bag of words for every paragraph
for paragraph in paragraphs:
    # Turns every paragraph into a list of pairs: (word index, word count)
    bagsOfWords.append(dictionary.doc2bow(paragraph))


# ------------
# 3. Retrieval Models
# ------------
# TF-IDF model using corpus (list of paragraphs)
tfidf_model = gensim.models.TfidfModel(bagsOfWords)
# Map Bags of Words into TF-IDF weights
tfidf_corpus = tfidf_model[bagsOfWords]
# Calculate MatrixSimilarity that let us calculate similarities between paragraph and queries
tfidf_matrixSimilarity = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# TF-IDF model using the TF-IDF weights (corpus)
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
# Map Baf of Words into LSI weights
lsi_corpus = lsi_model[bagsOfWords]
# Calculate MatrixSimilarity that let us calculate similarities between paragraph and queries
lsi_matrixSimilarity = gensim.similarities.MatrixSimilarity(lsi_corpus)

# Show 3 paragraphs, with all their words
print("[ First 3 LSI topics ]")
for topic in lsi_model.show_topics(num_topics=3):
    print(topic)


# ------------
# 4. Querying
# ------------
# Filter the input, stem it and convert to BOW (Bag of Words)
query = "What is the function of money?"
query = filterInput(query)
query = stem(query)
query = dictionary.doc2bow(query)

# Convert BOW to TF-IDF representation
tfidf_weights = tfidf_model[query]
# Report the TF-IDF weights
print("\n[ TF-IDF weights ]")
for i, weight in tfidf_weights:
    print(str(dictionary[i]) + ": " + str(weight) + ", ", end="")
print("\n")

# 3 most relevant paragraphs for query
doc2similarity = enumerate(tfidf_matrixSimilarity[tfidf_weights])
# Key is used to define what the sorting function is going to sort on
# Lambda kv: -kv[1] sorts by elements at position 1 in the array, and the - makes it be sorted reversed
doc2similarity = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
# Print 5 lines of each of the 3 most relevant paragraphs
printRelevantParagraphs(doc2similarity)

# Convert query TF-IDF representation into LSI-topics representation (weights)
lsi_query = lsi_model[query]
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]

# Print the top 3 topics
for topic, weight in topics:
    print("[ Topic " + str(topic) + " ]")
    print(lsi_model.show_topics()[topic])
print()

doc2similarity = enumerate(lsi_matrixSimilarity[lsi_query])
doc2similarity = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
# Print 5 lines of each of the 3 most relevant paragraphs
printRelevantParagraphs(doc2similarity)