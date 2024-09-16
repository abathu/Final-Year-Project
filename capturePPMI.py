import collections
import pickle
import typing
from typing import Any, List, Tuple, Union , Dict
from multiprocessing import Pool, cpu_count
from functools import partial

import spacy
from tqdm import tqdm
import re
import pandas as pd
from gensim.models.phrases import original_scorer
from gensim.models import Word2Vec

import numpy as np
# from numpy import dot, ndarray
from numpy.linalg import norm



def categorize_and_sort_gradient_words(gradient_data: Dict[str, List[float]]) -> Tuple[List[str], List[str]]:
    """
    Categorize and sort words based on their similarity score gradients into increasing or decreasing lists,
    sorted by the mean of their gradients.

    Args:
    gradient_data (Dict[str, List[float]]): Dictionary where each key is a word and the value is a list of gradients of similarity scores.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two sorted lists:
        - The first list contains words with increasing gradients sorted by the mean of gradients.
        - The second list contains words with decreasing gradients sorted by the mean of gradients.
    """
    increasing_gradients = []
    decreasing_gradients = []

    for word, gradients in gradient_data.items():
        if all(g > 0 for g in gradients):  # Check if all gradients are positive
            increasing_gradients.append((word, sum(gradients) / len(gradients)))
        elif all(g < 0 for g in gradients):  # Check if all gradients are negative
            decreasing_gradients.append((word, sum(gradients) / len(gradients)))

    # Sort the tuples by the mean of the gradients and extract the word for final list
    sorted_increasing_words = [word for word, _ in sorted(increasing_gradients, key=lambda x: x[1], reverse=True)]
    sorted_decreasing_words = [word for word, _ in sorted(decreasing_gradients, key=lambda x: x[1])]

    return sorted_increasing_words, sorted_decreasing_words



def categorize_gradient_words(gradient_data: Dict[str, List[float]]) -> Tuple[List[str], List[str]]:
    """
    Categorize words based on their similarity score gradients into increasing or decreasing lists.

    Args:
    gradient_data (Dict[str, List[float]]): Dictionary where each key is a word and the value is a list of gradients of similarity scores.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists:
        - The first list contains words with increasing gradients.
        - The second list contains words with decreasing gradients.
    """
    increasing_gradients = []
    decreasing_gradients = []

    for word, gradients in gradient_data.items():
        if all(g > 0 for g in gradients):  # Check if all gradients are positive
            increasing_gradients.append(word)
        elif all(g < 0 for g in gradients):  # Check if all gradients are negative
            decreasing_gradients.append(word)

    return increasing_gradients, decreasing_gradients



def calculate_similarity_gradient(similarities: List[float]) -> List[float]:
    """
    Calculate the gradient of similarity scores between consecutive elements.

    Args:
    similarities (List[float]): List of similarity scores for a word across different models or times.

    Returns:
    List[float]: List of gradients between consecutive similarity scores.
    """
    # Calculate the gradients between consecutive similarity scores
    gradients = [similarities[i] - similarities[i - 1] for i in range(1, len(similarities))]
    return gradients


def find_sim(words_vectors: Dict[str, List[np.ndarray]], words: List[str], centre_word_vector: List[np.ndarray]) -> Dict[str, List[float]]:
    """
    Find the similarity of vectors of words to corresponding center word vectors.

    Args:
    words_vectors (Dict[str, List[np.ndarray]]): Dictionary mapping words to lists of their vectors.
    words (List[str]): List of words to find similarities for.
    centre_word_vector (List[np.ndarray]): List of central word vectors, one for each word.

    Returns:
    Dict[str, List[float]]: Dictionary mapping words to their similarity scores with the center word vectors.
    """
    similarity_scores = {}
    for word in words:
        if word in words_vectors:
            vectors = words_vectors[word]
            word_similarity = [capture_cosine_similarity(vector, centre_word_vector[i]) for i, vector in enumerate(vectors)]
            similarity_scores[word] = word_similarity
        else:
            # Handling cases where the word might not be in the dictionary
            similarity_scores[word] = f'Word {word} not in dictionary'

    return similarity_scores



def find_word_vectors(target_words: List[str], models: List[Word2Vec]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Retrieve the word vectors for a list of target words across multiple Word2Vec models.

    Args:
        target_words (List[str]): List of words for which vectors are required.
        models (List[Word2Vec]): List of Word2Vec models from which vectors are to be retrieved.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary where keys are words and values are dictionaries.
            Each nested dictionary has model identifiers as keys and the corresponding word vectors as values.
    """
    words_similarities = {}
    # Iterate through each target word
    for word in target_words:
        model_vectors = [None]* len(models)
        # Retrieve vectors from each model
        for i, model in enumerate( models):
            try:
                # Get vector for the word from the current model
                vector = model.wv[word]
                model_vectors[i] = vector
            except KeyError:
                # Handle the case where the word is not in the model's vocabulary
                model_vectors[model] = None
        # Store vectors from all models for the current word
        words_similarities[word] = model_vectors

    return words_similarities



def normalize_and_find_mean(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Normalize each vector in a list and calculate the mean of these normalized vectors.

    Args:
    vectors (List[np.ndarray]): A list of numpy arrays (vectors).

    Returns:
    np.ndarray: The mean vector of the normalized input vectors.
    """
    # Normalize each vector: divide by its norm, avoid division by zero
    normalized_vectors = [vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else np.zeros_like(vector) for vector in vectors]

    # Compute the mean of the normalized vectors
    mean_vector = np.mean(normalized_vectors, axis=0)

    return mean_vector




def capture_cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors, ensuring both are NumPy arrays.

    Args:
    vector_1 (np.ndarray): First vector.
    vector_2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity between the two vectors. Returns None if inputs are not arrays or zero vectors.

    Raises:
    TypeError: If either input is not a NumPy array.
    """
    # Check if both inputs are numpy arrays
    if not isinstance(vector_1, np.ndarray) or not isinstance(vector_2, np.ndarray):
        raise TypeError("Both inputs must be numpy.ndarray.")

    # Check for zero vectors to avoid division by zero
    if np.all(vector_1 == 0) or np.all(vector_2 == 0):
        return 0.0  # Return 0.0 if either vector is a zero vector to indicate no similarity

    # Calculate the cosine similarity
    cosine_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return cosine_similarity





def get_sorted_positive_words_vector (file_path: str) -> List[Tuple[str, float]]:
    """
    Returns a sorted list of words with positive coefficients from a pickled file.

    Args:
    - file_path (str): The path to the pickled file containing coefficients and words.

    Returns:
    - List[Tuple[str, float]]: A list of tuples containing words and their corresponding positive coefficients, sorted by coefficient values in descending order.
    """
    # Load coefs_with_fns from the file
    with open(file_path, 'rb') as file:
        loaded_coefs_with_fns = pickle.load(file)

    # Filter positive coefficients and sort by coefficient value
    positive_words_sorted = sorted(
        [(word, coef) for coef, word in loaded_coefs_with_fns if coef > 0],
        key=lambda x: x[1],
        reverse=True
    )

    return positive_words_sorted


def get_sorted_positive_words (file_path: str) -> List[Tuple[str]]:
    """
    Returns a sorted list of words with positive coefficients from a pickled file.

    Args:
    - file_path (str): The path to the pickled file containing coefficients and words.

    Returns:
    - List[Tuple[str, float]]: A list of tuples containing words and their corresponding positive coefficients, sorted by coefficient values in descending order.
    """
    # Load coefs_with_fns from the file
    with open(file_path, 'rb') as file:
        loaded_coefs_with_fns = pickle.load(file)

    # Filter positive coefficients and sort by coefficient value
    positive_words_sorted = sorted(
        [word for coef, word in loaded_coefs_with_fns if coef > 0],
        key=lambda x: x[1],
        reverse=True
    )

    return positive_words_sorted




def calculate_npmi_score(pair, bigram_count, worda_count, wordb_count, len_vocab, min_count, corpus_word_count):
    # Calculate NPMI score
    npmi_score = original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count)
    return pair, npmi_score


class Corpus:
    def __init__(self, corpus: List[str], window_size: int = 10, min_freq: int = 30, n_processes=None, *,
                 word_list: List[str] = None, label: str = None):
        """
        parameter : corpus  --> untokenize corpus
                  label
        """
        self.window_size = window_size
        self.n_processes = n_processes if n_processes else cpu_count()
        self.label = label

        self.corpus = self._tokenizerWord(corpus)

        self.min_freq = min_freq
        # Flatten the list of tokenized sentences to a list of words
        flattened_list = [word for sentence in self.corpus for word in sentence]
        # Count the frequency of each token
        counter = collections.Counter(flattened_list)

        # Filter tokens by frequency
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        ## all tokens after thresholding
        self.idx_to_token = [token for token, freq in self.token_freqs if freq >= min_freq]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # Initialize a set to keep track of all unique tokens and meaning full
        nlp = spacy.load("en_core_web_sm")
        stop_words = nlp.Defaults.stop_words
        self.all_tokens_set = [token for token in self.token_to_idx if token not in stop_words]

        self._co_occurrences = {}

    @property
    def all_tokens(self):
        # Return a list of all unique tokens
        return list(self.all_tokens_set)

    def get_corpus(self):
        return self.corpus

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, None)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    def counting_occur(self):
        # Reset co_occurrences each time the function is called to avoid accumulating counts from multiple calls
        self._co_occurrences = {}

        for sentence in tqdm(self.corpus, desc="Counting co-occurrences"):
            # Iterate over each word in the sentence as the center of a window
            for center_idx, center_word in enumerate(sentence):
                if center_word in self.token_to_idx:
                    # Calculate the window range
                    start = max(center_idx - (self.window_size // 2), 0)
                    end = min(center_idx + (self.window_size // 2) + 1, len(sentence))

                    # Fetch unique words within the window excluding the center word
                    # and ensure each is counted once per window around the center word
                    counted_pairs = set()  # Track pairs already counted in this window to avoid double-counting
                    for i in range(start, end):
                        co_word = sentence[i]
                        if co_word in self.token_to_idx and co_word != center_word:
                            # Create an unordered pair as a frozenset for easy checking of duplicates
                            pair = frozenset([center_word, co_word])
                            if pair not in counted_pairs:
                                # Convert the frozenset back to a tuple for consistent indexing in the dictionary
                                pair_tuple = tuple(pair)
                                self._co_occurrences[pair_tuple] = self._co_occurrences.get(pair_tuple, 0) + 1
                                counted_pairs.add(pair)

    @staticmethod
    def worker_function(texts):
        nlp = spacy.load("en_core_web_sm")
        # Regex pattern to match words starting with '@' or '@gt', 'RT', and URLs
        at_pattern = re.compile(r'@gt\w+|@\w+')
        rt_pattern = re.compile(r'\bRT\b')
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        processed_texts = []
        for text in texts:
            # Remove matches for the patterns defined above
            text = at_pattern.sub('', text)
            text = rt_pattern.sub('', text)
            text = url_pattern.sub('', text)
            doc = nlp(text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
            processed_texts.append(tokens)
        return processed_texts

    def _tokenizerWord(self, corpus):
        chunksize = int(len(corpus) / self.n_processes) + 1
        chunks = [corpus[i:i + chunksize] for i in range(0, len(corpus), chunksize)]

        with Pool(self.n_processes) as pool:
            results = list(tqdm(pool.imap(Corpus.worker_function, chunks), total=len(chunks), desc="Tokenizing"))
        return [sentence for chunk in results for sentence in chunk]

    def get_co_occurrence_count(self, word_a, word_b):
        """Returns the co-occurrence count for a specific word pair."""
        """Since we calculate the window size twice for each word, we do divide by 2 here """
        pair = tuple(sorted([word_a, word_b]))
        return self._co_occurrences.get(pair, 0) // 2

    def calculate_npmi_vectors(self):
        len_vocab = len(self.idx_to_token)
        corpus_word_count = sum(count for word, count in self.token_freqs)
        word_counts = {word: count for word, count in self.token_freqs}

        self.npmi_vectors = {word: np.zeros(len_vocab) for word in self.idx_to_token}

        # Preparing the arguments for multiprocessing
        args = []
        for pair in self._co_occurrences.keys():
            worda, wordb = pair
            if worda in self.token_to_idx and wordb in self.token_to_idx:
                bigram_count = self.get_co_occurrence_count(worda, wordb)
                worda_count = word_counts.get(worda, 0)
                wordb_count = word_counts.get(wordb, 0)
                args.append((pair, bigram_count, worda_count, wordb_count, len_vocab, self.min_freq, corpus_word_count))
        # Using multiprocessing to calculate NPMI scores

        with Pool(self.n_processes) as pool:
            results = list(tqdm(pool.starmap(calculate_npmi_score, args), total=len(args), desc="Calculating NPMI"))

        # Update npmi_vectors with the results
        for pair, npmi_score in results:
            worda, wordb = pair
            index_a, index_b = self.token_to_idx[worda], self.token_to_idx[wordb]
            self.npmi_vectors[worda][index_b] = npmi_score
            self.npmi_vectors[wordb][index_a] = npmi_score


if __name__ == "__main__":
    corpus_path = r"/Users/chenyujie/Desktop/ClassifiedData/Reddit/2014_02_done.csv"
    df = pd.read_csv(corpus_path, header='infer')
    corpus = df['text'].tolist()
    test_corpus = Corpus(corpus)

    # for NPPMI
    test_corpus.counting_occur()
    test_corpus.calculate_npmi_vectors()
    nppmi_matrix = test_corpus.npmi_vectors
    word_you = nppmi_matrix['you']
