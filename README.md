# Search-Engine

# Authors
[ Aaron Aung, Jonathan Lin ]

# Description 
[ A web search engine built with pure front end tools without a framework. The web application queries the back-end program written in Python which performs machine learning algorithms such as cosine similarity and TFIDF calculations to return the best search results for users ]

# Tools Used
[ HTML, CSS, Bootstrap, jQuery, Python, Flask, scikit-learn, beautifulsoup, MongoDB ]

# How it works
*Processing State* <br/>
[ The python program processes a corpus of crawled web pages (not included in this repo becaues of size), and creates an inverted index with calculated TFIDF scores which gets inserted into a Mongo database ]

*Live Application* <br/>
[ The web application performs an AJAX on the middle-tier application built with Flask (a python library) that queries the database and returns results ranked based on proximity between the query and each data point (each search result) ] 
