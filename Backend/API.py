import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from tokenizer.Tokenizer import *
from src.Transformer import *
from config import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
src_tokenizer = None
tgt_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

activations = {}
decoder_activations = []
inputSeq = []
outputSeq = []
precalc = []
step_probabilities = []
attention_weights = {}

def get_activation(name):
    def hook(_, input_field, output):
        activations[name] = {
            "input": input_field[0].detach().cpu(),
            "output": output.detach().cpu()
        }
    return hook

def get_attention_hook(name):
    def hook(module, input, output):
        _, weights = output
        attention_weights[name] = weights
    return hook

def hookUp():
    for i, encoder in enumerate(model.encoder.encoderList):
        encoder.norm1.register_forward_hook(get_activation(f"encoder-norm-{i}-1"))
        encoder.norm2.register_forward_hook(get_activation(f"encoder-norm-{i}-2"))
        encoder.multiHeadedAttention.register_forward_hook(get_attention_hook(f"encoder-attn-{i}"))

    for i, decoder in enumerate(model.decoder.decoderList):
        decoder.norm1.register_forward_hook(get_activation(f"decoder-norm-{i}-1"))
        decoder.norm2.register_forward_hook(get_activation(f"decoder-norm-{i}-2"))
        decoder.norm3.register_forward_hook(get_activation(f"decoder-norm-{i}-3"))
        decoder.multiHeadedAttention1.register_forward_hook(get_attention_hook(f"decoder-self-attn-{i}"))
        decoder.multiHeadedAttention2.register_forward_hook(get_attention_hook(f"decoder-cross-attn-{i}"))

def getModel():
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    files = os.listdir(MODEL_FOLDER)
    
    if not files:
        path = hf_hub_download(
            repo_id=MODEL_URL,
            filename="model-0.pt",
            local_dir=MODEL_FOLDER
        )
    else:
        latest_file = sorted(files)[-1]
        path = os.path.join(MODEL_FOLDER, latest_file)
    
    m = initTransformer(D_MODEL, VOCAB_SIZE, HEADS, LAYERS, DROPOUT)
    checkpoint = torch.load(path, map_location=device)
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    m.load_state_dict(state_dict)
    
    m.to(device)
    m.eval()
    return m

def prepareSource(text, tokenizer):
    tokens = tokenizer.encode(text).ids
    seq = torch.cat([
        torch.tensor([tokenizer.token_to_id("[SOS]")]),
        torch.tensor(tokens),
        torch.tensor([tokenizer.token_to_id("[EOS]")]),
    ])
    paddings = MAX_SEQ_LEN - len(seq)
    if paddings > 0:
        seq = torch.cat([seq, torch.tensor([tokenizer.token_to_id("[PAD]")] * paddings)])
    else:
        seq = seq[:MAX_SEQ_LEN]
    src_mask = (seq != tokenizer.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(1).to(device)
    return seq.unsqueeze(0).to(device), src_mask

def greedyPredict(src_tokens, src_mask):
    global decoder_activations, outputSeq, step_probabilities

    decoder_activations = []
    step_probabilities = []
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

            probs = torch.softmax(predictions, dim=-1)
            step_probabilities.append(probs.cpu())

        step_snapshot = {name: {k: v.clone() for k, v in val.items()}
                         for name, val in activations.items() if "decoder" in name}
        decoder_activations.append(step_snapshot)

        output_id = torch.argmax(predictions[0, 0]).item()
        decoder_input[0, index] = output_id
        generated_ids.append(output_id)

        if output_id == tgt_tokenizer.token_to_id("[EOS]"):
            break
    outputSeq = [tgt_tokenizer.id_to_token(idx) for idx in generated_ids]

@app.get("/translate/")
async def translate(text: str):
    try:
        src_tokens, src_mask = prepareSource(text, src_tokenizer)
        global inputSeq
        inputSeq = src_tokens[0].tolist()
        greedyPredict(src_tokens, src_mask)
        return " ".join(outputSeq)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attention/")
async def get_attention(layer_id: int, type_l: str):
    try:
        key = f"{type_l}-attn-{layer_id}"
        if type_l == "encoder": key = f"encoder-attn-{layer_id}"
        weights = attention_weights.get(key).detach().cpu()
        if weights is None:
            raise HTTPException(status_code=404, detail="Weights not found.")
        src_tokens = [src_tokenizer.id_to_token(i) for i in inputSeq if src_tokenizer.id_to_token(i) != "[PAD]"]
        tgt_tokens = outputSeq
        if type_l == "encoder":
            rows, cols = src_tokens, src_tokens
        elif type_l == "decoder-self":
            rows, cols = tgt_tokens, tgt_tokens
        elif type_l == "decoder-cross":
            rows, cols = tgt_tokens, src_tokens
        else:
            raise HTTPException(status_code=400, detail="Invalid attention type")
        sliced_weights = weights[0, :, :len(rows), :len(cols)]
        return {
            "weights": sliced_weights.tolist(),
            "src_tokens": cols,
            "tgt_tokens": rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attentionMeta/")
async def get_attention_metadata(layer_id: int, type_l: str):
    try:
        key = f"{type_l}-attn-{layer_id}"
        if type_l == "encoder": key = f"encoder-attn-{layer_id}"
        weights = attention_weights.get(key)
        if weights is None:
            raise HTTPException(status_code=404, detail="Weights not found.")
        return weights.shape[1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/softmax/")
async def get_softmax():
    try:
        if not step_probabilities:
            raise HTTPException(status_code=404, detail="No translation performed yet.")
        output_data = []
        for t, probs in enumerate(step_probabilities):
            dist = probs.squeeze()
            top_probs, top_indices = torch.topk(dist, k=5)
            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                candidates.append({
                    "token": tgt_tokenizer.id_to_token(idx.item()),
                    "prob": prob.item()
                })
            output_data.append({
                "timestep": t,
                "subsentence": outputSeq[:t + 1],
                "candidates": candidates,
                "actual_selection": outputSeq[t + 1] if t + 1 < len(outputSeq) else None
            })
        return output_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/norm/")
async def norm(type_l: int, id_l: int, num: int, token: int, step: int = 0):
    try:
        if type_l == 0:
            act = activations.get(f"encoder-norm-{id_l}-{num}")
            layer = model.encoder.encoderList[id_l]
            word = src_tokenizer.id_to_token(inputSeq[token])
        else:
            step_idx = min(step, len(decoder_activations) - 1)
            act = decoder_activations[step_idx].get(f"decoder-norm-{id_l}-{num}")
            layer = model.decoder.decoderList[id_l]
            word = outputSeq[token] if token <= step else "[PAD]"
        if not act: raise HTTPException(status_code=404)
        norm_mod = getattr(layer, f"norm{num}")
        return {
            "input": act["input"][0, token].tolist(),
            "output": act["output"][0, token].tolist(),
            "word": word,
            "gamma": round(norm_mod.alpha[token].item(), 5),
            "beta": round(norm_mod.beta[token].item(), 5)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedding/by-word/")
async def get_word_embedding_by_word(word: str, lang: str):
    try:
        tokenizer = src_tokenizer if lang == "en" else tgt_tokenizer
        embedder = model.src_embedding if lang == "en" else model.tgt_embedding
        token_id = tokenizer.encode(word.lower().strip()).ids[0]
        with torch.no_grad():
            vector = embedder(torch.tensor([token_id], device=device))[0].tolist()
        return {"word": tokenizer.id_to_token(token_id), "token_id": token_id, "vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedding/by-id/")
async def get_word_embedding_by_id(token_id: int, lang: str):
    try:
        tokenizer = src_tokenizer if lang == "en" else tgt_tokenizer
        embedder = model.src_embedding if lang == "en" else model.tgt_embedding
        target_word = tokenizer.id_to_token(token_id)
        with torch.no_grad():
            vector = embedder(torch.tensor([token_id], device=device))[0].tolist()
        return {"word": target_word, "token_id": token_id, "dimensions": len(vector), "vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedForward/")
async def get_ffn_graph(id_l: int, type_l: int):
    try:
        layer = model.encoder.encoderList[id_l] if type_l == 0 else model.decoder.decoderList[id_l]
        w1_tensor = layer.feedForward.l1.weight.detach().cpu()
        w2_tensor = layer.feedForward.l2.weight.detach().cpu()
        total_in, total_hid, total_out = w1_tensor.shape[1], w1_tensor.shape[0], w2_tensor.shape[0]
        n_in, n_hid, n_out = 10, 20, 10
        return {
            "w1": w1_tensor[:n_hid, :n_in].tolist(),
            "w2": w2_tensor[:n_out, :n_hid].tolist(),
            "totals": {"in": total_in, "hid": total_hid, "out": total_out},
            "labels": {"in": [f"i_{i}" for i in range(n_in)], "hid": [f"h_{i}" for i in range(n_hid)], "out": [f"o_{i}" for i in range(n_out)]}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedding/gallery/")
async def get_word_embedding_gallery(page: int, page_size: int, lang: str):
    try:
        tokenizer = src_tokenizer if lang == "en" else tgt_tokenizer
        embedder = model.src_embedding if lang == "en" else model.tgt_embedding
        start, end = page * page_size, (page + 1) * page_size
        weights = embedder.embedding.weight[start:end] * (embedder.dmodel ** 0.5)
        return [{"word": tokenizer.id_to_token(i), "token_id": i, "vector": v.tolist()}
                for i, v in zip(range(start, end), weights)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positional/matrix")
async def get_positional_matrix():
    return precalc

@app.get("/N/")
async def n_layers():
    return len(model.encoder.encoderList)

@app.get("/normlen/")
async def norm_len():
    return {"n": len(inputSeq), "maxStep": len(outputSeq)}

@app.get("/lastInput/")
async def last_input():
    try:
        filtered_input = " ".join([src_tokenizer.id_to_token(i) for i in inputSeq if i not in [
            src_tokenizer.token_to_id(j) for j in ["[PAD]", "[SOS]", "[EOS]"]]])
        return {"input": filtered_input, "output": " ".join(outputSeq)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projection/")
async def get_projection_graph():
    try:
        weights = model.projection.linear.weight.detach().cpu()
        n_in, n_out = 20, 30
        return {
            "weights": weights[:n_out, :n_in].tolist(),
            "totals": {"in": weights.shape[1], "out": weights.shape[0]},
            "input_labels": [f"dim_{i}" for i in range(n_in)],
            "output_labels": [tgt_tokenizer.id_to_token(i) for i in range(n_out)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        CAUSAL_MASK = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, device=device)).bool()
        src_tokenizer = getOrBuildTokenizer(ENGLISH_TOKENIZER_PATH, getDataset(DATASET_PATH), "en")
        tgt_tokenizer = getOrBuildTokenizer(FRENCH_TOKENIZER_PATH, getDataset(DATASET_PATH), "fr")
        model = getModel()
        hookUp()
        with torch.no_grad():
            temp_pos = torch.zeros(1, MAX_SEQ_LEN, D_MODEL, device=device)
            precalc = model.positionalEncoding(temp_pos)[0].transpose(0, 1).detach().cpu().tolist()
        print(f"Server ready on {device}!")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Fatal error during initialization: {e}")