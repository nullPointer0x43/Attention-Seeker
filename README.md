# Transformer-Vis: A From-Scratch Transformer Implementation and Attention Visualizer
## 1. Overview:
Transformer-Vis is a comprehensive Transformer Model trained for Translation. It is a from-scratch implementation of the Transformer architecture as proposed in "Attention Is All You Need." 

Unlike standard implementations that utilize high-level abstractions, this project is a ground-up reconstruction of the core architecture. This includes the manual implementation of Multi-Head Attention, Positional Encodings, and specialized masking logic. A React-based frontend provides real-time visualization of attention weights, mapping the linguistic alignment between source and target sequences during inference.

The core idea of this project, was to get accustomed to the architecture and the mathematics behind transformer models, as a stepping stone to further research on topics such as State Space Machines (SSMs), Mamba, etc.

## 2. Table of Contents:
Architecture Components

Multi-Head Attention

Positional Encoding

Encoder and Decoder Stacks

Dataset and Preprocessing

Training Methodology

Deployment and Docker Integration

Future Roadmap

External Resources and References

## 3. Architecture Description and Visualizations:
### 3.1. Embedding Model:
#### **Technical Explanations:**
The Embedding Layer is the first layer, which converts word tokens to discrete vectors. In the simplest terms it is a 2-dimensional look-up matrix. It has dimensions of `vocab_size` x `d_model`. 

Implicitly pytorch converts a token id number (lets say 4) to a one hot vector `[0, 0, 0, 0, 1, 0, 0 0, ...]`. Once this one hot vector is multiplied by the embedding matrix, it outputs a d_model sized vector. Logically the vector for the token with `id = n` is the nth row of the embedding matrix. This vector is then multiplied by $\sqrt{d_{model}}$ to scale the variance of the vector upto the order of magnitude of 1. (complicated math stuff... present in my notes for those interested)

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

![Embedding-visualization](/documentation-images/embedding.png)

### 3.2 Positional Encoding:
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
