# Skip-Gram Word2Vec with Negative Sampling

## Relevant papers
Efficient Estimation of Word Representations in Vector Space (Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean) - https://arxiv.org/abs/1301.3781
Distributed Representations of Words and Phrases and their Compositionality (Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean) - https://arxiv.org/abs/1310.4546

Bonus paper, very helpful to better understqnd the backprobagation happening with Negative Sampling: word2vec Parameter Learning Explained (Xin Rong) https://arxiv.org/pdf/1411.2738.pdf

## Word2Vec

Word vectors are the representation of words (part of a vocabulary) by vectors. These vectors can be based on frequency i.e. the number of times they appear in co-occurence with other words of the vocabulary. In this case the word vector dimension is (1, vocab_size) and the co-occurence matrix's dim is (vocab_size, vicab_size)

To answer these limitations(n² space), Tomas Mikolov, Kai Chen, Greg Corrado & Jeffrey Dean in their paper "Efficient Estimation of Word Representations in Vector Space" resorted to word embeddings where words are no longer represented by co-occurences (i.e. counts) but by vectors of fixed size (1, n_embedding).

However, updating all the weights of this (vocab_size, embedding_size) matrix at each training step was inefficient and could use further improvement. This is where negative sampling comes in handy.

To explain negative sampling I will use the case of Skip-Gram Word2Vec which is the one built in this jupyter notebook.


## Skip-Gram

The Skip-gram model represents learning the context given a center word (as opposed to the CBOW model). For instance in the sentence we might try to learn what context occurs at the same time as the word "fox" and in our corpus we might have the sentence "The quick brown fox jumps over the lazy dog". Therefore we will try to learn that "quick" and "brown" occur close to the word "fox".

![Skip-Gram](https://github.com/shawn-lab-ml/ml_from_scratch/blob/master/nlp/Word2Vec/skipgram.png)


## Negative Sampling

For Negative Sampling we will do the same as in the example above. However, we will add one step: sample (according to frequency) other words and tell our model that these words do not co-occur.

This transforms the model into a binary classification problem where we attempt to optimize the following cost function :
![Negative Sampling Cost Function](https://github.com/shawn-lab-ml/ml_from_scratch/blob/master/nlp/Word2Vec/Word2vec%20NS%20cost%20function.PNG)

This can be summarized as attempting to get the left sigmoid's input equal to 1 i.e. the context words have a strong co-occurence with the center word and the right sigmoid's input equal to 1 as well but as: sigmoid(-a) = 1 - sigmoid(a), we are actually trying to learn that the negative samples do not occur often in the context of the center word. By doing so we just have to update the matrices (embedding matrix and dense layer) for the negative samples and context words rather than update the whole matrix
