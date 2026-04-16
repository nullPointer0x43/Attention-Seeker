# Attention-Seeker: A From-Scratch Transformer Implementation and Attention Visualizer
## 1. Overview:
Transformer-Vis is a comprehensive Transformer Model trained for Translation. It is a from-scratch implementation of the Transformer architecture as proposed in "Attention Is All You Need." 

Unlike standard implementations that utilize high-level abstractions, this project is a ground-up reconstruction of the core architecture. This includes the manual implementation of Multi-Head Attention, Positional Encodings, and specialized masking logic. A React-based frontend provides real-time visualization of attention weights, mapping the linguistic alignment between source and target sequences during inference.

The core idea of this project, was to get accustomed to the architecture and the mathematics behind transformer models, as a stepping stone to further research on topics such as State Space Machines (SSMs), Mamba, etc.

## 2. Table of Contents:

1. **[Overview](#1-overview)**
2. **[Table of Contents](#2-table-of-contents)**
3. **[Architecture Description and Visualizations](#3-architecture-description-and-visualizations)**
    - 3.1. **[Tokenization Model](#31-tokenization-model)**
    - 3.2 **[Embedding Model](#32-embedding-model)**
    - 3.3 **[Positional Encoding](#33-positional-encoding)**
    - 3.4 **[Attention Mechanism](#34-attention-mechanism)**
        - 3.4.1 **[What is Self and Cross Attention?](#what-is-self-and-cross-attention)**
        - 3.4.2 **[How Much Head Do You Need?](#how-much-head-do-you-need)**
        - 3.4.3 **[What's Masking?](#whats-masking)**
        - 3.4.4 **[Putting everything together](#putting-everything-together)**
        - 3.4.5 **[Computation and Complexity](#computation-and-complexity)**
    - 3.5 **[Feed Forward and Projection Layer](#35-feed-forward-and-projection-layer)**
    - 3.6 **[Normalization Layer](#36-normalization-layer)**
    - 3.7 **[Softmax](#37-softmax)**
        - 3.7.1 **[Auto-regression](#auto-regression)**
    - 3.8 **[Connecting All the parts](#38-connecting-all-the-parts)**
4. **[Training Details](#4-training-details)**
    - 4.1 **[Training Data](#training-data)**
    - 4.2 **[Training Procedure](#training-procedure)**
5. **[Tech Stack](#5-tech-stack)**
    - 5.1 **[Front-end: React & CSS](#front-end-react--css)**
    - 5.2 **[Backend: FastAPI & PyTorch](#backend-fastapi--pytorch)**
    - 5.3 **[Serving & Containerization: Nginx & Docker](#serving--containerization-nginx--docker)**
6. **[How To Run](#6-how-to-run)**
    - 6.1 **[Docker Compose](#61-docker-compose-recommended)**
    - 6.2 **[Manual Setup](#62-manual-setup)**
7. **[Additional Links](#7-additional-links)**
8. **[Credits and References](#8-credits-and-references)**

## 3. Architecture Description and Visualizations:
### 3.1. Tokenization Model:
#### **Technical Explanations:**
Tokenization is the first step in the entire pipeline. This layer converts raw string (words and sentences) into a string of numbers called `token_ids`. 

There are 3 types of tokenizers I explored and coded up:

**1. Word Level Tokenizers:**

This is the tokenizer that the entire pipeline runs on currently. It works by spliting sentences on whitespaces and punctuations.

* **Training Procedure:** Given a corpus of words, it finds the top `n` most frequent words where `n = vocab_size`. It then stores and assigns these words specific `ID's`. These words form the vocabulary of the tokenizer.
* **Drawback-1:** If the model encounters a word it hasn't seen during training (like a plural form or a typo), it assigns it to a generic `[UNK]` (Unknown) token, losing all semantic meaning.
* **Drawback-2:** It cannot group and cluster according to similar root words, hence has a very large vocabulary size. (E.g. it classifies win and winning as 2 seperate tokens)
* **Then why was this chosen?** This model was the simplest and fastest to train, hence was chosen due to computational limitations.

**Implementation:**
```python
from data.Dataset import *

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os

def getOrBuildTokenizer(tokenizerPath, ds, lang):
    if lang not in ['en', 'fr']: raise ValueError('lang must be in ["en", "fr"]')

    if not os.path.exists(tokenizerPath):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
                                             vocab_size=VOCAB_SIZE, min_frequency=2)
        print(ds)
        sentences = ds[lang]
        tokenizer.train_from_iterator(sentences, tokenizer_trainer)

        tokenizer.save(tokenizerPath)
    else:
        tokenizer = Tokenizer.from_file(tokenizerPath)

    return tokenizer
```

**2. BPE:**

BPE is the a relatively simpler version of the tokenizers used by most modern Transformers. It breaks down rare words into meaningful sub-units (e.g., "smartest" = "smart" + "est").

* **Training Procedure:** It starts with a base vocabulary of individual characters. It then iteratively identifies the most frequently occurring adjacent pair of tokens in the corpus and merges them into a single new token. This process continues for a fixed number of `merges` until the desired `vocab_size` is reached.
* **Drawback:** While it reduces the frequency of [UNK] tokens, it can still fail if it encounters a rare Unicode character (like a specific emoji or obscure script) that wasn't in the initial character-level base vocabulary.

**Implementation:**
```python
# Full code in TokenizerCustom.py
def train(self, text, verbose=False):
    merged = {}
    text = [int(i) for i in text.encode("utf-8")]

    for i in range(self.vocabSize - 256):
        stats = self.getStats(text)
        pair = max(stats, key=stats.get)
        text = self.merge(text, pair, i + 256)
        merged[pair] = i + 256

        if verbose: print(pair, "merged to", i + 256)

    self.merges = merged

    self.vocab = {i: bytes([i]) for i in range(256)}

    for (p0, p1), idx in self.merges.items():
        self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
```

**3. BPE with Regex (GPT-2 Style):**
* **Training Procedure:** This adds a pre-segmentation step using a specific Regex pattern. Before merging, the text is split into chunks based on categories (words, numbers, or contractions like 's, 't, 're). BPE merging is then strictly restricted to happen within these chunks, never across them.
* **Advantage:** By preventing merges between distinct categories (like a letter and a punctuation mark), the tokenizer maintains better structural integrity of the language.

**Implementation:**
```python
#Regex Expression used:
# 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

def train(self, text, verbose=False):
    import regex as re
    merged = {}
    text = [[int(j) for j in i.encode("utf-8")] for i in re.findall(self.regex, text)]

    for i in range(self.vocabSize - 256):
        stats = self.getStats(text)
        pair = max(stats, key=stats.get)
        text = self.merge(text, pair, i + 256)
        merged[pair] = i + 256

        if verbose: print(pair, "merged to", i + 256)

    self.merges = merged

    self.vocab = {i: bytes([i]) for i in range(256)}

    for (p0, p1), idx in self.merges.items():
        self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
```

### 3.2. Embedding Model:
#### **Technical Explanations:**
The Embedding Layer is the first layer, which converts word tokens to discrete vectors. In the simplest terms it is a 2-dimensional look-up matrix. It has dimensions of `vocab_size` x `d_model`. 

Implicitly pytorch converts a token id number (lets say 4) to a one hot vector `[0, 0, 0, 0, 1, 0, 0 0, ...]`. Once this one hot vector is multiplied by the embedding matrix, it outputs a d_model sized vector. Logically the vector for the token with `id = n` is the $n^{th}$ row of the embedding matrix. This vector is then multiplied by $\sqrt{d_{model}}$ to scale the variance of the vector upto the order of magnitude of 1. (complicated math stuff... present in my notes for those interested)

**Implementation:**
```python
class embedding(nn.Module):
    def __init__(self, vocabSize, dmodel):
        super().__init__()

        self.vocabSize = vocabSize
        self.dmodel = dmodel

        self.embedding = nn.Embedding(vocabSize, dmodel)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel)
```

#### **Front-end Visualization:**
The user can interact, find and explore the vector embeddings of different words (tokens). 

* Each vector consists of `d_model` number of stripes, with varying color to represent the different values.

* A dedicated gallery provides a scrollable interface to compare the embeddings of the entire vocabulary.

### 3.3 Positional Encoding:
#### **Technical Explanations:**
The Positional Encoding layer, is a completely deterministic mathematical layer. It adds information about the position of the token within the sentence / sequence. 

* It utilizes the `sine` and `cosine` functions. Since these functions have are bounded (range of `[-1, 1]`) and are periodic in nature, they make for perfect candidates for encoding positional information.

* Each position in a sequence is mapped to a vector of size `d_model` composed of wave functions with varying frequencies. This generates a unique encoding for every position.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Implementation:**
```python
class positionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])
```

#### **Front-end Visualization:**
A similar vector visualisation to that used int he embedding layer, displays the vector created for a given position in a sequence.

The gradient diagram below shows a continuous spectrum of the positional encodings. 

* The `X-Axis` represents the position in the sequence
* The `Y-Axis` represents items in each of the vector

Hence a slice on pixel value `x=n` will give you the positional encoding for a token in the $n^{th}$ position.

### 3.4 Attention Mechanism:
#### **Tehnical Explanations:**
The attention mechanism is the heart of the transformer architecture. In simple words it finds semantic relations between words within a sentence. 

It uses 3 matrices, to transform the input vector into a set of 3 vectors.
* $W_k$ = `Weight matrix for keys`
* $W_v$ = `Weight matrix for values`
* $W_q$ = `Weight matrix for queries`

On multiplying the input vector `I` with these matrices we get the following:
* $I \times W_k = K$ (The Key matrix)
* $I \times W_q = Q$ (The Query matrix)
* $I \times W_v = V$ (The Value matrix)

Intuitively the matrices can be thought of as:
* The `Value Vector` containins a summary of the semantic value of the input vector. 
* The `Query Vector` converts the input vector into a query asking every other token in the sequence: "How much are you related to me?"
* The `Key Vector` answers the query of the `Query Vector` 

Mathematically speaking the dot product: 

$$Score = Q_a \cdot K^T_b$$

represents how much a key from a $token_b$ corresponds to the query of $token_a$.

In the code implementation instead of providing a single input vector, provision has been made to pass 3 seperate vectors. This is relevant and important for **Cross Attention** as discussed in the next section.

```python
def forward(self, q, k, v, mask=None):
    ...
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    ...
    attention_scores = query @ key.transpose(-2, -1)
```

#### **WHAT IS SELF AND CROSS ATTENTION?**

Now when we pass a single input vector to be converted into the Value, Query and Key Vectors, it is called **Self Attention**. It can be thought of as finding relationships within the same sentence.

However what makes it truly powerful, is if the model is able to look at the word it is currently translating and ask the original English sentence: "Which part of the original context should I be focusing on right now to get this specific word right?". 

This exact problem is solved using **Cross Attention**. In Cross Attention the input for the Query vector comes from the decoder block (the section forming the new translated sequence) while the inputs for the Key and Value vector come from the encoder block (the section which condenses and extracts context from the original sequence). 

```python
#Self Attention
self.multiHeadedAttention(x, x, x, tgt_mask)

#Cross Attention
self.multiHeadedAttention(x, context, context, tgt_mask)
```

#### **HOW MUCH HEAD DO YOU NEED?**

As the name of the module suggests, the architecture proposed in the *"All you need is Attention"* paper is a **Multi Headed Attention Architecture**.

* While a single attention head can learn relationships, it is limited by only being able to focus on one "type" of relationship at a time. The Multi-Head architecture solves this by running multiple attention mechanisms in parallel.
* Instead of running the whole attention block on the entire embedding vector multiple times, it is split the vector along its embedding dimension.
* The `d_model` (e.g. 512) is split into $k$ smaller heads (e.g. 8 heads of 64 dimensions each).
* This "split" approach ensures that the total number of parameters and the computational cost are equivalent to a single-head attention with full dimensionality, while significantly increasing the model's ability to learn complex patterns in parallel.
* The results from all heads are concatenated to produce the final output.

```python
# (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k)
query = query.view(query.shape[0], query.shape[1], self.k, self.d_k).transpose(1, 2)

# (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k) --> (batch, k, d_k, seq)
key = key.view(key.shape[0], key.shape[1], self.k, self.d_k).transpose(1, 2)

# (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k)
value = value.view(value.shape[0], value.shape[1], self.k, self.d_k).transpose(1, 2)
```

#### **WHAT'S MASKING?**

**Masking** is a strategy employed to prevent the attention model from attending to unnecessary tokens (e.g. `[PAD]`) during computation.

There are two distinct types of maskings:
* Padding Masks: In the model, all the input sequences of irregular sizes are padded with a `[PAD]` token to have a final sequence of length `Max Sequence Length`. These Padding tokens need not be considered while calculating attention. Hence these columns and rows are masked.
* Causal Masks: This is used exclusively in the Decoder. During training, the model has access to the entire target sentence. To simulate a real-world scenario where the model must predict the next word without knowing the future, a "triangular" mask is applied. This prevents $token_n$ from looking at $token_{n+1}$ or any subsequent tokens.

**Mathematical Implementation:** Masking is achieved by adding a value of $-\infty$ (or a very large negative number) to the attention scores before the softmax layer.

#### **PUTTING EVERYTHING TOGETHER:**

The final Mathematical formula for attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + \text{Mask}}{\sqrt{d_k}}\right)V$$

**Dimensional Analysis:**
| Variable | Name | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| **$Q$** | Query | $L \times d_{k}$ | Represents $L$ tokens, each with a search vector of size $d_k$. |
| **$K^T$** | Key (Transposed) | $d_{k} \times L$ | The transposed Key matrix, for dot-product calculation. |
| **$V$** | Value | $L \times d_{v}$ | The semantic content for each token to be weighted by attention. |
| **Mask** | Attention Mask | $L \times L$ | Applied to $QK^T$ to nullify padding or future tokens ($-\infty$). |
| **Weights** | Softmax Output | $L \times L$ | The probability distribution defining the relationship between all tokens. |
| **Output** | Final Context | $L \times d_{v}$ | The weighted sum of Values; the new representation for each token. |

#### **COMPUTATION AND COMPLEXITY:**

| Operation | Equation | Dimensions | Complexity ($O$) |
| :--- | :--- | :--- | :--- |
| **Linear Projections** | $I \cdot W_{q,k,v}$ | $(L \times d_{model}) \cdot (d_{model} \times d_k)$ | $O(L \cdot d_{model} \cdot d_k)$ |
| **Attention Scores** | $Q \cdot K^T$ | $(L \times d_k) \cdot (d_k \times L)$ | $O(L^2 \cdot d_k)$ |
| **Mask Addition** | $Score + M$ | $(L \times L) + (L \times L)$ | $O(L^2)$ |
| **Weighted Sum** | $Weights \cdot V$ | $(L \times L) \cdot (L \times d_k)$ | $O(L^2 \cdot d_k)$ |

Hence the total time Complexity boils down to:

$$O(L \cdot d_{model} \cdot d_k) + O(L^2 \cdot d_k) + O(L^2) + O(L^2 \cdot d_k) \approx O(L^2)$$

Thus we can see why people say a transformer has quadratic complexity. *(P.S. i dont expect random ppl to be talking about the time complexity of transformer models)*

**Implementation**
```python
def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)

        # (batch, k, seq, d_k) @ (batch, k, d_k, seq) --> (batch, k, seq, seq)
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e6)

        attention_scores = torch.softmax(attention_scores, dim = -1)

        attention_scores = self.dropout(attention_scores)

        # (batch, k, seq, seq) @ (batch, k, seq, d_k) --> (batch, k, seq, d_k)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask=None):
        # (batch, seq, dmodel) --> (batch, seq, dmodel)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.k, self.d_k).transpose(1, 2)

        # (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k) --> (batch, k, d_k, seq)
        key = key.view(key.shape[0], key.shape[1], self.k, self.d_k).transpose(1, 2)

        # (batch, seq, dmodel) --> (batch, seq, k, d_k) --> (batch, k, seq, d_k)
        value = value.view(value.shape[0], value.shape[1], self.k, self.d_k).transpose(1, 2)

        x, scores = self.attention(query, key, value, mask)

        # (batch, k, seq, d_k) -> (batch, seq, k, d_k) --> (batch, seq, dmodel)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)
        x = self.w_o(x)

        return x, scores
```

#### **Front-end Visualization:**

**Key Features:**
* **Head-Specific Heatmaps:** Allows users to cycle through all $k$ attention heads. This visually demonstrates the specialized feature extraction performed by the Multi-Head architecture.
* By hovering over any token in the input or output blocks, the interface **isolates the specific connections for that token.**
* The magnitude of the attention weights is represented by the **thickness and opacity** of the connecting lines, providing an intuitive "map" of the model's semantic focus.

### 3.5 Feed Forward and Projection Layer:
#### **Technical Explanations:**
* All the feed-forward blocks in the network operate on each token embedding independently and identically.
* To allow the model to learn more complex non-linear patterns, the model typically expands the vector to a higher dimension before projecting it back down.
* Thus it generally consists of two layers, one for expanding and one for projecting back down.
* each layer can be described mathematically as in a normal **Multi-Layer-Perceptron**:

$$F(x) = \sigma(x \times W + b)$$

where:
* $\sigma$ is the activation function (in this case ReLU).
* $W$ is the weight matrix of dimension $(d_{model} \times d_{hidden})$ and $(d_{hiddden} \times d_{model})$ for the two layers respectively.
* $b$ is the bias matrix

The **Projection Layer** is a special type of feed-forward neural network which is positioned after the final output of the decoder block. Its purpose is to project the high-dimensional internal representations ($d_{model}$) back into the massive space of the `Vocabulary Size`.

Each of these values are meant to signify the likelihood of the correspond token being the next token.

**Implementation:**
```python
class feedForward(nn.Module):
    def __init__(self, d_model, dropout, d_ff =  2048):
        super().__init__()

        self.dmodel = d_model
        self.d_ff = d_ff

        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.l2(x)

        return x

class projectionLayer(nn.Module):
    def __init__(self, d_model, vocabSize):
        super().__init__()

        self.linear = nn.Linear(d_model, vocabSize)

    def forward(self, x):
        return self.linear(x)
```

#### **Front-end Visualization:**
* Instead of raw numbers, weights are visualized using a diverging colors (Red and Blue).
* By hovering over any edge or node in the network graph, the UI reveals the precise floating-point value of that weight. This helps probe the actual matrices ($W_1, W_2$).

### 3.6 Normalization Layer:
#### **Technical Explanations:**
Most machine learning models work better with normalized inputs, i.e. inputs that have a `mean = 0` and a `standard deviation = 1`.

The model has 2 extra learnable parameters (distinct for each position in sequence) to tweak and learn the most optimal distribution. The formula for normalization is as follows:

$$x' = \gamma \frac{x-\mu}{\sigma + \epsilon} + \beta$$

where:
* $\gamma$ is the multiplicative coeffecient
* $\beta$ is the additive coeffecient
* $\mu$ is the mean of the vector
* $\sigma$ is the standard deviation of the vector
* $\epsilon$ is some small value to prevent division by 0

$\gamma$ helps scale the distribution while $\beta$ shifts them along the `x-axis`.
```python
# Gamma == alpha
class layerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.dmodel = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x = (x - torch.mean(x, dim=2, keepdim=True)) / (torch.std(x, dim=2, keepdim=True) + self.eps)
        x = self.alpha * x + self.beta
        return x
```

#### **Front-end Visualization:**
The front-end provides an excellent visualisation for the compressive effect as the vector gets normalized.

* Instead of viewing embeddings as static lists of numbers, the interface converts them into **discrete frequency histograms.** This allows users to observe the "shape" of the data as it flows through the network.
* The interface displays the learned **$\gamma$ (Scaling) and $\beta$ (Shifting)** parameters in real-time. This reveals how the model is mathematically re-centering and re-scaling the data.

### 3.7 Softmax:
#### **Technical Explanations:**
As discussed earlier the final projection layer gives a vector of `vocab_size` for each length in the sequence. This is essentially the probability of different tokens being in a particular position in the sequence.

We are interested in predicting the next token. Hence if we are predicting the $n^{th}$ token we look at the $n^{th}$ vector from the final outputs. But these are raw numbers, how do we convert them to a continuous probability distribution. That is done using the soft-max function:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Implementation:**
```python
return torch.softmax(x, dim=-1)
```

This function coverts each individual value in the vector into a probability.

#### **Auto-regression:**

Unlike the Encoder, which processes the entire source sentence simultaneously, the Decoder is Autoregressive. 

* This means it generates the output one token at a time, using its own previous predictions as input for the next step.
* The decoder starts with a simple completely padded string with only a `[SOS]` token at the $0^{th}$ index. 
* It runs an inference loops and checks the output probabilities with respect this current incomplete translation (just the `[SOS]` token).
* It then appends this new token to the current translation.
* This process repeats till it finally predicts an `[EOS]` token.

**Implementation**
```python
def greedyPredict(src_tokens, src_mask):
    outputSeq = ["[SOS]"]

    with torch.no_grad():
        encoder_output = model.encode(src_tokens, src_mask)

    generated_ids = [tgt_tokenizer.token_to_id("[SOS]")]
    sos_id = tgt_tokenizer.token_to_id("[SOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")

    decoder_input = torch.full((1, MAX_SEQ_LEN), tgt_tokenizer.token_to_id("[PAD]"), dtype=torch.long, device=device)
    decoder_input[0, 0] = tgt_tokenizer.token_to_id("[SOS]")

    for index in range(1, MAX_SEQ_LEN):
        current_causal = CAUSAL_MASK[:index, :index].unsqueeze(0).unsqueeze(1)
        tgt_mask = (decoder_input[:, :index] != pad_id).unsqueeze(1).unsqueeze(2) & current_causal

        with torch.no_grad():
            decoder_output = model.decode(decoder_input[:, :index], encoder_output, tgt_mask, src_mask)

            last_token_feat = decoder_output[:, -1:, :]
            predictions = model.project(last_token_feat)

        output_id = torch.argmax(predictions[0, 0]).item()
        decoder_input[0, index] = output_id
        generated_ids.append(output_id)

        if output_id == tgt_tokenizer.token_to_id("[EOS]"):
            break
```

#### **Front-end Visualization:**
The front-end for this layer is an interactive display of the top-5 most likely tokens at every time-step.

* Each candidate is paired with a real-time histogram representing its relative probability. This allows users to see if the model is highly confident in its choice (a single tall bar) or if it is, hesitant between multiple options (several medium-sized bars).
* A scrollable timeline allows users to navigate back and forth through the decoding sequence

### 3.8 Connecting all the parts:
The entire transformer architecture works by stacking all these layers and blocks. Here is the chronological flow of a single inference step:

1. **Source Processing (The Input):**

    1.1. The raw English sentence enters the system:

    1.2 **Tokenization:** The string is split into  `token IDs` using the **Word-level tokenizer**.
    
    1.3 **Embedding:** `IDs` are mapped to a $d_{model}$ vector space.
    
    1.4 **Positional Encoding:** Sinusoidal signals are added to give the model a sense of position of each token in the sequence.

2. **The Encoder Stack (The Contextualizer):** 

    2.1. The processed input passes through $N$ Encoding Blocks (typically $N=6$ in the original architecture). Each block performs:
    
    2.2. **Multi-Head Self-Attention:** Allowing tokens to talk to each other and find relations within the input sequence.
    
    2.3. **Add & Norm:** A residual connection followed by Layer Normalization.
    
    2.4. **Feed-Forward Network:** A MLP for internal processing.
    
    2.5 **Add & Norm:** Final stabilizing layer for the block.

    The output of the $N^{th}$ block is a set of Vectors which intuitively represent the entire source sentence.

3. **Decoder Processing & Stack (The Generator)**

    The Decoder handles the partially predicted output (starting with `[SOS]`)

    3.1. **Initial Processing:** Just like the source, the target tokens are embedded and positionally encoded.
    
    3.2. **The Decoding Stack:** Like the Encoder, this consists of $N$ Decoding Blocks ($N=6$). However, with slight changes and a few extra steps.
    
    3.3. **Masked Self-Attention:** Tokens only look at past predicted tokens. (Masking) 
    
    3.4. **Add & Norm**
    
    3.5. **Cross-Attention:** The "Bridge" where the Decoder queries the Encoder’s Memory Vectors.
    
    3.6. **Add & Norm**
    
    3.7. **Feed-Forward Network**
    
    3.8. **Add & Norm**
    
4. **The Final Decision (The Output)**

    4.1. **Projection Layer:** The Decoder’s final output is projected from $d_{model}$ back to the full `vocab_size`.
    
    4.2. **Softmax:** Probabilities are calculated, and the most likely token is selected as the next word in the translation.

| Stage | Operations | Depth ($N$) |
| :--- | :--- | :--- |
| **Source Processing** | Tokenize $\rightarrow$ Embed $\rightarrow$ Pos-Encode | - |
| **Encoder Stack** | Self-Attention $\rightarrow$ Add&Norm $\rightarrow$ FFN $\rightarrow$ Add&Norm | 6 Layers |
| **Decoder Processing** | Target-Tokenize $\rightarrow$ Embed $\rightarrow$ Pos-Encode | - |
| **Decoder Stack** | Masked-Attention $\rightarrow$ Cross-Attention $\rightarrow$ FFN | 6 Layers |
| **Final Output** | Linear Projection $\rightarrow$ Softmax | - |

## 4. Training Details:
### Training Data
1. Dataset: Helsinki-NLP/opus_books (English-French subset)

2. Size: ~127,000 sentence pairs

### Training Procedure
1. Hardware: NVIDIA GeForce RTX 4060 Laptop GPU

2. Training Time: ~2.5 Hours

3. Optimizer: Adam

4. Learning Rate: 1e-4 (Fixed)

5. Batch Size: 4

6. Epochs: 1 (Proof of Concept)

7. Loss Function: Cross-Entropy Loss

## 5. Tech-Stack:
### Front-end: React & CSS
The interface is built with React 20, using component-based architecture to mirror the modularity of the Transformer itself.

* **State Management:** We use custom hooks and context providers to manage the heavy data flow of multi-dimensional tensors arriving from the backend.

* **Visualization Suite:** Basic CSS used for the responsive and aesthetic dashboard layout.

* **Dynamic SVG rendering** and **CSS Transitions** animate attention blocks and intermediary data flow.

### Backend: FastAPI & PyTorch
The backend of the application is a FastAPI server that acts as a wrapper around the PyTorch model.

* **Inference Engine:** Upon receiving an input string, the backend triggers the PyTorch forward pass, extracting not just the prediction, but all the intermediate data for visualization (Attention weights, LayerNorm distributions, etc.).

* **Asynchronous Processing:** FastAPI’s async capabilities ensure that the UI remains responsive while the model predicts and relays the data.

### Serving & Containerization: Nginx & Docker
To ensure the application is stable and easy to deploy:

* **Nginx:** Acts as a Reverse Proxy, handling incoming traffic and routing it to the FastAPI backend. It provides an extra layer of security and can be used to serve static frontend assets efficiently.

* **Containerization:** The entire stack—Frontend, Backend, and Nginx—is containerized via Docker Compose. This allows the project to run in a consistent environment regardless of the host OS, ensuring reproducability.

## 6. How To Run:

### 6.1 Docker Compose: (Recommended)

This method pulls pre-configured images for the React frontend, FastAPI backend, and Nginx proxy, ensuring the environment exactly matches the development setup.

Prerequisites: Docker and Docker Compose installed.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/transformer-viz.git
cd transformer-viz
```

**2. Launch the Stack**
```bash
docker-compose up -d
```

**3. Access the Application**
* **Frontend:** Open `http://localhost:3000`

* **Backend API:** all API Endpoints available on `http://localhost:8000`

### 6.2 Manual Setup:
I used Python 3.11, since I was unable to find later versions (3.12+) which have stable support for the CUDA-compatible binaries required for PyTorch inference on NVIDIA GPUs.

#### **Backend Setup:**
**1. Environment Initialization:**
```bash
cd backend
# Ensure you are using Python 3.11
python3.11 -m venv venv
source venv/bin/activate
```

**2. Install all Libraries and Dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Run the API Server:**
```bash
python API.py
```

#### **Frontend Setup:**
**1. Install & Launch:**
```bash
cd ./Frontend
npm install
npm run dev
```

## 7. Additional Links:

The model weights are automatically pulled by the scripts from huggingface. Or can be manually downloaded from the following:

https://huggingface.co/kinjal2/Attention-Seeker-Model

The docker images for both the frontend and backend can be found on docker-hub:

* https://hub.docker.com/r/kinjal1234/transformer-backend (Backend)
* https://hub.docker.com/r/kinjal1234/transformer-frontend (Frontend)

## 8. Credits and References:
The following are the resources, papers and videos I used in this project.

**Attention Is All You Need Paper**
* https://arxiv.org/abs/1706.03762

**Transformer Architecture & Attention Mechanism**
* https://jalammar.github.io/illustrated-transformer/
* https://youtu.be/eMlx5fFNoYc?si=GPBzM3aBUDGsAvAV
* https://youtu.be/zduSFxRajkE?si=8wZQAjPR53PLIAaI
* https://youtu.be/KMHkbXzHn7s?si=jO-3RDlifbpQDd_l
* https://youtu.be/9-Jl0dxWQs8?si=85jA5CC_nhNvcmWm

**Model Implementations Referred: (BERT / NLP)**
* https://github.com/jessevig/bertviz
* https://youtu.be/ISNdQcPhsts?si=ywsYOr7O4SU31k-j

**General Deep Learning / Project Guidance**
* https://youtu.be/GeoQBNNqIbM?si=7pfntFo6vNqiZxui
* https://youtu.be/G6D9cBaLViA?si=EjicAirawQWDoaaY

**UI / Design Assets**
* https://www.flaticon.com/