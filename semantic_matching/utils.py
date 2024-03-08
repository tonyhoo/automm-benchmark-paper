import numpy as np
import torch
import sys
from sentence_transformers import util


def compute_sims(img_embs, cap_embs, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = img_embs[im_start:im_end]
                ca = cap_embs[ca_start:ca_end]
                sim = util.cos_sim(im, ca)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims

def t2i(num_text, num_img, sims, img_dup=5):
    assert num_text == num_img * img_dup
    ranks = np.zeros(num_text)
    for index in range(num_img):
        for i in range(img_dup):
            inds = np.argsort(sims[img_dup * index + i])[::-1]
            ranks[img_dup * index + i] = np.where(inds == index)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    rmean = (r1 + r5 + r10) / 3

    return (r1, r5, r10, rmean)

def i2t(num_text, num_img, sims, img_dup=5):
    assert num_text == num_img * img_dup
    ranks = np.zeros(num_img)
    for index in range(num_img):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(img_dup * index, img_dup * index + img_dup, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    rmean = (r1 + r5 + r10) / 3

    return (r1, r5, r10, rmean)

def truncate_sentence(sentence, tokenizer):
    """
    Truncate a sentence to fit the CLIP max token limit (77 tokens including the
    starting and ending tokens).
    
    Args:
        sentence(string): The sentence to truncate.
        tokenizer(CLIPTokenizer): Rretrained CLIP tokenizer.
    """
    
    cur_sentence = sentence
    tokens = tokenizer.encode(cur_sentence)
    
    if len(tokens) > 77:
        # Skip the starting token, only include 75 tokens
        truncated_tokens = tokens[1:76]
        cur_sentence = tokenizer.decode(truncated_tokens)
        
        # Recursive call here, because the encode(decode()) can have different result
        return truncate_sentence(cur_sentence, tokenizer)
    
    else:
        return cur_sentence