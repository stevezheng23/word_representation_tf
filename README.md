# Language Model
Language modeling is a task that assigns probabilities to sequences of words or various linguistic units (e.g. char, subword, sentence, etc.). Language modeling is one of the most important problem in modern natural language processing (NLP) and it's used in many NLP applications (e.g. speech recognition, machine translation, text summarization, spell correction, auto-completion, etc.). In the past few years, neural approaches have achieved better results than traditional statistical approaches on many language model benchmarks. Moreover, recent work has shown language model pre-training can improve many NLP tasks in different ways, including feature-based strategies (e.g. ELMo, etc.) and fine-tuning strategies (e.g. OpenAI GPT, BERT, etc.), or even in zero-shot setting (e.g. OpenAI GPT-2, etc.).

<p align="center"><img src="/language_model/document/language_model.example.png" width=600></p>
<p align="center"><i>Figure 1: An example of auto-completion powered by language modeling</i></p>

## Setting
* Python 3.6.6
* Tensorflow 1.12
* NumPy 1.15.4
* NLTK 3.3

## DataSet
* [Wikipedia corpus](https://www.corpusdata.org/wikipedia.asp) contains about 2 billion words of text from a 2014 dump of the Wikipedia (about 4.4 million pages). As far as we are aware, our Wikipedia full-text data is the only version available from a recent copy of Wikipedia.
* [BooksCorpus](http://yknzhu.wixsite.com/mbweb): Books are a rich source of both fine-grained information, how a character, an object or a scene looks like, as well as high-level semantics, what someone is thinking, feeling and how these states evolve through a story. This work aims to align books to their movie releases in order to provide rich descriptive explanations for visual content that go semantically far beyond the captions available in current datasets.
* [One Billion Word](http://www.statmt.org/lm-benchmark/) benchmark is targeted to make available a standard training and test setup for language modeling experiments. This benchmark contains almost one billion words of training data, and it's aiming to help researcher to quickly evaluate novel their language modeling techniques, and to easily compare the contributions when combined with other advanced techniques.
* [GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Usage
* Preprocess data
```bash
# convert raw data
python preprocess/convert_data.py --dataset wikipedia --input_dir data/wikipedia/raw --output_dir data/wikipedia/processed --min_seq_len 0 --max_seq_len 512
# prepare vocab & embed files
python prepare_resource.py \
--input_dir data/wikipedia/processed --max_word_size 512 --max_char_size 16 \
--full_embedding_file data/glove/glove.840B.300d.txt --word_embedding_file data/wikipedia/resource/lm.word.embed --word_embed_dim 300 \
--word_vocab_file data/wikipedia/resource/lm.word.vocab --word_vocab_size 100000 \
--char_vocab_file data/wikipedia/resource/lm.char.vocab --char_vocab_size 1000
```
* Run experiment
```bash
# run experiment in train + eval mode
python language_model_run.py --mode train_eval --config config/config_lm_template.xxx.json
# run experiment in train only mode
python language_model_run.py --mode train --config config/config_lm_template.xxx.json
# run experiment in eval only mode
python language_model_run.py --mode eval --config config/config_lm_template.xxx.json
```
* Encode text
```bash
# encode text as ELMo vector
python language_model_run.py --mode encode --config config/config_lm_template.xxx.json
```
* Search hyper-parameter
```bash
# random search hyper-parameters
python hparam_search.py --base-config config/config_lm_template.xxx.json --search-config config/config_search_template.xxx.json --num-group 10 --random-seed 100 --output-dir config/search
```
* Visualize summary
```bash
# visualize summary via tensorboard
tensorboard --logdir=output
```

## Model
### Bi-directional Language Model (biLM)
Given a sequence, the bi-directional language model computes the probability of the sequence forward,
<p align="center"><img src="/language_model/document/bilm.eqn.fwd.gif" width=300, align="center"></p>
<!-- p \left ( t_{1}, t_{2}, ..., t_{N} \right ) = \prod_{k=1}^{N} p \left ( t_{k} | t_{1}, t_{2}, ..., t_{k-1} \right ) -->
then it runs over the sequence in reverse order to compute the probability of the sequence,
<p align="center"><img src="/language_model/document/bilm.eqn.bwd.gif" width=300></p>
<!-- p \left ( t_{1}, t_{2}, ..., t_{N} \right ) = \prod_{k=1}^{N} p \left ( t_{k} | t_{k+1}, t_{k+2}, ..., t_{N} \right ) -->
the sequence first goes through a shared embedding layer, then is modeled by multi-layer RNN (e.g. LSTM, GRU, etc.) in both directions and finally softmax normalization is applied to get probabilities,
<p align="center"><img src="/language_model/document/bilm.architecture.png" width=500></p>
<p align="center"><i>Figure 2: bi-directional language model architecture (source: <a href="https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html">Generalized Language Models</a>)</i></p>
the model is trained by jointly minimizing the negative log likelihood of the forward and backward directions,
<p align="center"><img src="/language_model/document/bilm.eqn.loss.gif" width=600><br /></p>
<!-- L \left ( \Theta \right ) = - \sum_{k=1}^{N} \left ( log p \left ( t_{k} | t_{1}, t_{2}, ..., t_{k-1} ; \Theta_{e}, \overset{ \rightarrow }{ \Theta }_{RNN}, \Theta_{s} \right ) + log p \left ( t_{k} | t_{k+1}, t_{k+2}, ..., t_{N} ; \Theta_{e}, \overset{ \leftarrow }{ \Theta }_{RNN}, \Theta_{s} \right ) \right ) -->

## Reference
* Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matthew Gardner, Christopher T Clark, Kenton Lee,
and Luke S. Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) [2018]
* Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [2018]
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805) [2018]
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [2019]
