from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup, Comment
from nltk.stem import SnowballStemmer
from urllib.parse import urlparse
import pprint
import json
import time
import re
import os


class Indexer:
    def __init__(self):
        self.file = open("WEBPAGES_RAW/bookkeeping.json", "r")
        self.pp = pprint.PrettyPrinter(indent=4)
        self.book = json.load(self.file)
        self.forward_index = {}
        self.inverted_index = {}

    @staticmethod
    # builds the path of a given directory and file number
    def build_path(dir_num, file_num):
        return "WEBPAGES_RAW/" + dir_num + "/" + file_num

    @staticmethod
    def strip_html(html):
        cleaned = " ".join(html.replace("\n", " ").strip().split())
        return cleaned

    @staticmethod
    def soup_clean(soup):
        # remove scripts
        for script in soup.findAll("script"):
            script.extract()

        # remove linked styles
        styles = soup.findAll("link", attrs={"type": "text/css"})
        for style in styles:
            style.extract()

        # remove on page styles
        for style in soup.findAll("style"):
            style.extract()

        # remove commented html
        comments = soup.findAll(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

    @staticmethod
    # return True and False based on whether we should "download" it
    def is_valid(url):
        if len(url) > 600:
            return False

        parsed = urlparse(url)
        try:
            return ".ics.uci.edu" in parsed.path and not re.match(".*\.(css|js|bmp|gif|jpe?g|ico"
                                    + "|png|tiff?|pdf|zip|rar|txt|py|java"
                                    + "|cpp|doc|docx|xls|xlsx|bin)$", parsed.path.lower())

        except TypeError:
            print("TypeError for ", parsed)
            return False

    # given an html string, generate its set of tokens
    def tokenize_html(self, html_str):
        soup = BeautifulSoup(html_str, "html.parser")
        self.soup_clean(soup)
        doc = soup.get_text().lower()
        tokenized = [tok for tok in re.split('[^a-z0-9]', doc) if len(tok) > 0]
        return tokenized

    # returns all html documents in corpus
    def get_documents(self):
        documents = {}
        invalid = 0
        for loc, url in self.book.items():
            if self.is_valid(url):
                dir_num, file_num = loc.split("/")
                html_str = self.get_html(dir_num, file_num)
                soup = BeautifulSoup(html_str, "html.parser")
                self.soup_clean(soup)
                documents[loc] = soup.get_text()
            else:
                invalid += 1

        print("> Number of invalid documents: " + str(invalid))
        return documents

    # returns html string given a directory and file number
    def get_html(self, dir_num, file_num):
        path = self.build_path(dir_num, file_num)
        file = open(path, "r")
        html = file.read()
        file.close()
        return html

    # adds {document, [tokens]} pair for every document
    def build_forward_index(self):
        for loc, url in self.book.items():
            if self.is_valid(url):
                print("Working on document", c)
                dir_num, file_num = loc.split("/")
                html_str = self.get_html(dir_num, file_num)
                tokens = self.tokenize_html(html_str)
                self.forward_index[loc] = tokens

    # generates a cleaned set of webpages
    def generate_clean_webpages(self):
        try:
            orig_dir = os.getcwd()
            os.mkdir("NEWEST_WP_CLEAN")
            os.chdir("NEWEST_WP_CLEAN")
            for i in range(75):
                os.mkdir(str(i))
            os.chdir(orig_dir)
            print("> Folders generated")
        except OSError:
            print("OSError")

        for loc, url in self.book.items():
            if self.is_valid(url):
                dir_num, file_num = loc.split("/")
                html_raw = self.get_html(dir_num, file_num)
                html_str = bytes(html_raw, 'utf-8').decode('utf-8', 'ignore')
                soup = BeautifulSoup(html_str, "html.parser")
                self.soup_clean(soup)

                path = "NEWEST_WP_CLEAN/" + str(dir_num) + "/"
                file = open(path + str(file_num), "w")
                clean_text = soup.get_text().encode("ascii", errors="ignore").decode()
                file.write(self.strip_html(clean_text))
                file.close()

    # grabs all the webpages and builds the inverted index
    def build_inverted_index(self):
        print("> Building Inverted Index")
        s = time.time()
        documents = self.get_documents()
        num_docs = len(documents)
        e = time.time()
        print("> Number of documents: " + str(num_docs))
        print("> Time for gathering documents: " + str(e-s))

        def tokenizer(doc):
            ss = SnowballStemmer('english')
            return[ss.stem(token) for token in re.split('[^a-z0-9]', doc) if len(token) > 0]

        s = time.time()
        tfidf = TfidfVectorizer(norm=None, use_idf=True, lowercase=True, smooth_idf=False,
                                sublinear_tf=True, tokenizer=tokenizer, stop_words='english')
        rep = tfidf.fit_transform(documents.values())
        tokens = tfidf.get_feature_names()
        idf = tfidf.idf_

        e = time.time()
        print("\n> TF-IDF Vectorization Complete")
        print("> Time for vectorization: " + str(e-s))
        index = 0
        alphanum = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

        print("\n> Constructing Index")
        start = time.time()
        document_locs = list(documents.keys())
        rep_array = rep.toarray()

        for i, tok in enumerate(tokens):  # for each token
            # if we have moved onto the next letter of the alphabet,
            # write to file the index of the previous letter
            if not tok.startswith(alphanum[index]):
                self.write_to_json(alphanum[index])
                self.inverted_index = {}
                index += 1

            self.inverted_index[tok] = [idf[i] - 1, {}]
            for k in range(num_docs):  # for each document
                tfidf = rep_array[k, i]
                if tfidf != 0:
                    self.inverted_index[tok][1][document_locs[k]] = tfidf

            def sort_key(item): return item[1]
            self.inverted_index[tok][1] = sorted(self.inverted_index[tok][1].items(), key=sort_key, reverse=True)

        self.write_to_json(alphanum[index])

        end = time.time()
        print("> Total time for index construction: " + str(end - start))
        print("\n> Number of unique terms: " + str(len(self.inverted_index)))

    def search_forward_index(self, token):
        document_list = []
        for document_loc, token_set in self.forward_index.items():
            if token in token_set:
                document_list.append(document_loc)
        return document_list

    def print_forward_index(self):
        self.pp.pprint(self.forward_index)

    def print_inverted_index(self):
        self.pp.pprint(self.inverted_index)

    def search_inverted_index(self, query, count):
        if query not in self.inverted_index:
            print("Cannot find search term")
            return None

        index = self.inverted_index[query][1]
        most_relevant = []

        c = 0
        for rel in index:
            most_relevant.append(rel)
            c += 1
            if c == count:
                break

        return most_relevant

    def lookup_url(self, loc):
        return self.book[loc]

    def write_to_json(self, name):
        json_index = open(name.upper() + "_inverted_index.json", "w")
        json.dump(self.inverted_index, json_index)
        json_index.close()
        print("\n" + name.upper() + " Written to JSON!")


def main():
    indexer = Indexer()
    indexer.build_inverted_index()
    indexer.generate_clean_webpages()


if __name__ == "__main__":
    main()
