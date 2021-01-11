# Skip-Gram Word2Vec with Negative Sampling

## Relevant papers
Efficient Estimation of Word Representations in Vector Space (Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean) - https://arxiv.org/abs/1301.3781
Distributed Representations of Words and Phrases and their Compositionality (Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean) - https://arxiv.org/abs/1310.4546

## Word2Vec

Word vectors are the representation of words (part of a vocabulary) by vectors. These vectors can be based on frequency i.e. the number of times they appear in co-occurence with other words of the vocabulary. In this case the word vector dimension is (1, vocab_size) and the co-occurence matrix's dim is (vocab_size, vicab_size)

To answer these limitations(n² space), Tomas Mikolov, Kai Chen, Greg Corrado & Jeffrey Dean in their paper "Efficient Estimation of Word Representations in Vector Space" resorted to word embeddings where words are no longer represented by co-occurences (i.e. counts) but by vectors of fixed size (1, n_embedding).

However, updating all the weights of this (vocab_size, embedding_size) matrix at each training step was inefficient and could use further improvement. This is where negative sampling comes in handy.

To explain negative sampling I will use the case of Skip-Gram Word2Vec which is the one built in this jupyter notebook.


## Skip-Gram

The Skip-gram model represents learning the context given a center word (as opposed to the CBOW model). For instance in the sentence we might try to learn what context occurs at the same time as the word "fox" and in our corpus we might have the sentence "The quick brown fox jumps over the lazy dog". Therefore we will try to learn that "quick" and "brown" occur close to the word "fox".


## Negative Sampling

For Negative Sampling we will do the same as in the example above. However, we will add one step: sample (according to frequency) other words and tell our model that these words do not co-occur.
