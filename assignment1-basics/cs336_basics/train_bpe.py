import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable
import regex as re
from dataclasses import dataclass

# relationships among all these data form and dataclasses
# data form:
# bpe -> "Byte pair encoding" , so the process is about handling bytes
# string is got at first, transforming it into bytes form is needed by encode("utf-8")
# token_id is the int by which we can fetch the corresponding token(bytes) from the vocab

# dataclasses
# vacab: the dict with which we can fetch the token by token_id
# merges: the list storing the merged bytes, built to help the vocab
# indices: only store the token_id, make it easy to merge because it only needs adjacent token_id


def merge(indices:list[int], pair:tuple[int,int], new_index:int)->list[int]:
    
    # given the most frequent pairs and get the new indices
    # indices is about the index of bytes in the vocabulary, merge functions 
    # as updating the new index of the merged tokens
    new_indices = []

    i = 0
    while i in range(len(indices)):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)
            i+=2# skipping two index because they are merged already
        else:
            new_indices.append(indices[i])
            i+=1 # with for loop, i in for loop won't be updated
         
    return new_indices

def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes,...],int]:
    # remove the special tokens and do pre-tokenization

    frequency_table: dict[tuple[bytes,...],int] = {}# initialize the dict

    # PAT given by the pdf help pretokenize the ordinary text
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # remove the special tokens first by re.split, need to form the specialtokens into a pattern
    special_pat = "|".join(re.escape(tok) for tok in special_tokens) # "|" means or in pattern and re.escape functions by tranforming a normal string into a pattern
    
    if special_pat != "": # when the special_pat is not "", remove it
        for part in re.split(special_pat,text):# remove the special tokens by re.split and get the ordinary text
            if part == "":
                continue
            else:
                for match in re.finditer(PAT,part): # return each item of the iterator
                    pretoken = match.group()
                    key = tuple(bytes([k]) for k in pretoken.encode('utf-8')) # transfer pretoken:str into key:tuple[bytes,...] and [k] means take k to be a byte int rather than a int
                    if key in frequency_table:
                        frequency_table[key] += 1
                    else:
                        frequency_table[key] = 1
    else: #avoid splitting text by ""
        parts = [text]
        for part in parts:
            if part == "":
                continue
            else:
                for match in re.finditer(PAT,part): # return each item of the iterator
                    pretoken = match.group()
                    key = tuple(bytes([k]) for k in pretoken.encode('utf-8'))
                    if key in frequency_table:
                        frequency_table[key] += 1
                    else:
                        frequency_table[key] = 1

    return frequency_table
# @dataclass(frozen=True)
# class bpeParams:
#     vocab: dict[int,bytes]
#     merges: list[tuple[bytes,bytes]]



# class BPETokenizer(Tokenizer):
#     """BPE tokenizer given a set of merges and a vocabulary."""
#     def __init__(self, params: bpeParams):
#         self.params = params
#     def encode(self, string: str) -> list[int]:
#         indices = list(map(int, string.encode("utf-8")))  
#         # Note: this is a very slow implementation
#         for pair, new_index in self.params.merges.items():  
#             indices = merge(indices, pair, new_index)  
#         return indices
#     def decode(self, indices: list[int]) -> str:
#         bytes_list = list(map(self.params.vocab.get, indices))  
#         string = b"".join(bytes_list).decode("utf-8")  
#         return string

# This is a very basic implementation of BPE training, which is not optimized for speed or memory.
# from multiprocessing import Pool
# from functools import partial
# wait for multiprocessing to be covered in class before trying to parallelize this
def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:

    with open(input_path,encoding='utf-8') as f:
        string = f.read()# get the text
    
    freq_table = pretokenize(string,special_tokens)
    # pretokenize the text and get the dict
    # need to change the code below the ensure they all accept the freq_table dict

    indices = list(map(int,string.encode('utf-8')))# encoding the text into bytes and then map it into int
    # need to be clarified about the function of input_path
    merges: list[tuple[bytes,bytes]] = []
    vocab : dict[int,bytes] = {x:bytes([x]) for x in range(256)}  
    #initialize the vocab by utf-8 and dict comprehension
    #dict comprehension: x:bytes([x]) while x is the key and bytes([x]) the value

    i = 0
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8")
    # append the special tokens to the vocab

    i = 0
    v = len(vocab)
    n = vocab_size - v # it changes so it cannot be used directly
    for i in range(n):
        # merge the frequncy pairs by n times
        pairs = countAdjacent(indices)
        pair = max(pairs,key=lambda k: (pairs[k], vocab[k[0]] + vocab[k[1]]))# comparing the bytes rather than the token id
        # return the most frequent one and break the tie by loxic order by lambda method

        new_index = v+i
        indices = merge(indices,pair,new_index)
        # gain the new_indices by the merge function

        merges.append((vocab[pair[0]],vocab[pair[1]]))
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        # update the bpeParameters
        # pair: tuple(int,int). need to fetch the bytes for the merges

    return vocab,merges


def countAdjacent(indices:list[int])->dict[tuple[int,int],int]:
    i = 0
    pairs = defaultdict(int)
    for i in range(len(indices)-1):
        pairs[(indices[i],indices[i+1])] += 1
        
    return dict(pairs) # transform pairs from defaultdict to dict


