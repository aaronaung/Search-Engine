from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup, Comment
from nltk.stem import SnowballStemmer
from collections import Counter
from pymongo import MongoClient
from flask_cors import CORS
from flask import request
from flask import Flask
import scipy.spatial as sp
import string
import json
import re

app = Flask(__name__)
CORS(app)

file = open("NEW_WP_CLEAN/bookkeeping.json")
book = json.loads(file.read())
cap = 500

def tokenizer(doc):
	ss = SnowballStemmer('english')
	return [ss.stem(token) for token in re.split('[^a-z0-9]', doc) if len(token) > 0]

def search(query_tokens, collection):
	# Search through inverted_index collection from the db
	# Parses mongo response and return a tuple([querytok-idf], [documents-per-querytok])
	documents = []
	idfs = [] # per token
	for token in query_tokens:
		query_dict = {token: 1, "_id": 0}
		filtered = list(filter(None, list(collection.find({}, query_dict))))
		if len(filtered) > 0:
			idfs.append(filtered[0][token][0])
		else:
			idfs.append(0)

		docs = [] if len(filtered) == 0 else filtered[0][token][1]
		documents.append(docs[:cap])
	return (idfs, documents)

def construct_tfidf_dictionary(doclist): # Doclist is a list of documents for each token
	# Maps document location to a tfidf vector (representing tfidf scores of each token)
	# Returns the map
	dictionary = {}
	for tok_index, documents in enumerate(doclist):
		for loc, tfidf_val in documents:
			tfidf_vector = [0] * len(doclist)
			tfidf_vector[tok_index] = tfidf_val
			if loc not in dictionary:
				dictionary[loc] = tfidf_vector
			else:
				dictionary[loc][tok_index] = tfidf_val
	return dictionary

def construct_query_tfidf(query_tokens, idfs):
	# Returns a tfidf score matrix for the query
	counts = Counter(query_tokens)
	tfidf_vector = []
	for i, token in enumerate(query_tokens):
		tf = counts[token] / len(query_tokens)
		tfidf = tf * idfs[i]
		tfidf_vector.append(tfidf)
	return [tfidf_vector] # this makes it a 1x3 matrix

def build_path(dir_num, file_num):
	return "NEW_WP_CLEAN/" + dir_num + "/" + file_num

def get_html(loc):
	dir_num, file_num = loc.split("/")
	path = build_path(dir_num, file_num)
	file = open(path, encoding="utf8")
	html = file.read()
	file.close()
	return html

def append_relevant_text(found, query, html):
	start_index = found.start(0)
	end_index = start_index + len(query)
	tail = (" ".join(html[end_index + 50:end_index + 100].split()[:-1])).strip()
	return "<b>" + html[start_index:end_index] + "</b>" + html[end_index:end_index + 50] + tail + " ..."

def relevant(doc_loc, query):
	# Returns first 150 characters + 100 characters closest to the searched tokens
	pre_stemmed = [tok.lower() for tok in query.split()]
	tokens = tokenizer(query)
	html = get_html(doc_loc)

	found = re.search(query, html, re.IGNORECASE)
	display_text = " ".join(html[:150].split()[:-1]) + " ..."
	if found:
		display_text += append_relevant_text(found, query, html)
	else:
		for i, tok in enumerate(tokens):
			found = re.search(tok, html, re.IGNORECASE)
			if found:
				end_index = found.start(0) + len(pre_stemmed[i])
				if html[found.start(0):end_index].lower() == pre_stemmed[i]:
					tok = pre_stemmed[i]
				display_text += append_relevant_text(found, tok, html) + " "
	return display_text

@app.route("/search", methods=['POST'])
def start():
	# Connect to mongo client
	client = MongoClient('mongodb://localhost:27017/')
	db = client["CS121"]
	collection = db["alphanumeric_index"]
	query = request.form['query'].lower()
	query_tokens = tokenizer(query)

	# the search result format: (query idfs, documents)
	query_idfs, documents = search(query_tokens, collection) # documents represents a list of document-list per token
	ranked_docs = []

	if len(query_tokens) > 1:
		# Construct matrices for cosine_similarity calcuation (only required for multiword search)
		query_tfidf = construct_query_tfidf(query_tokens, query_idfs) # query tfidf matrix
		tfidf_map = construct_tfidf_dictionary(documents)
		docs_tfidf = list(tfidf_map.values()) # tfidfx map matrix
		docs_loc = list(tfidf_map.keys())  # document location list

		if len(docs_loc) == 0:
			return json.dumps(ranked_docs)

		# calculate cosine similary between the query matrix and the document matrix
		similarity = cosine_similarity(query_tfidf, docs_tfidf, 'cosine')
		for i, loc in enumerate(docs_loc):
			relevant_texts = relevant(loc, query)
			ranked_docs.append([book[loc], similarity[0][i], relevant_texts])
		ranked_docs.sort(key=lambda sublist: sublist[1], reverse=True)
	else:
		# if it's a single-word query, return a list ranked by tfidf values
		for loc, tfidf in documents[0]:
			relevant_texts = relevant(loc, query)
			ranked_docs.append([book[loc], tfidf, relevant_texts])

	return json.dumps(ranked_docs)

if __name__ == "__main__":
 	app.run()
