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

# need to be updated for updating the freq_table rather than indices
def merge(freq_table:dict[tuple[int,...],int], pair:tuple[int,int], new_index:int)->dict[tuple[int,...],int]:
    
    # given the most frequent pairs and get the new indices
    # indices is about the index of bytes in the vocabulary, merge functions 
    # as updating the new index of the merged tokens
    new_freq_table = {}
    for key in freq_table:
        i = 0
        new_key = ()
        while i in range(len(key)):# just used to update the key and after the while loop, new_freq_table should be updated
            if i+1 in range(len(key)) and key[i] == pair[0] and key[i+1] == pair[1]:
                new_key = new_key + (new_index,) #get the new key by connecting them,but the key right of the pair shouldn't be connected because there may still have matched pair
                i+=2

            else:
                new_key = new_key + (key[i],)
                i+=1

        new_freq_table[new_key] = freq_table[key]
    return new_freq_table

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
def freq_table_trans(freq_table:dict[tuple[bytes,...],int])->dict[tuple[int,...],int]:
    new_freq_table = {
        tuple(x[0] for x in key): count
        for key, count in freq_table.items()
    }
    
    return new_freq_table


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

    freq_table = freq_table_trans(freq_table)
    # need to be changed into dict[tuple(int,...),int] to better fit the whole process
    

    #indices = list(map(int,string.encode('utf-8')))# encoding the text into bytes and then map it into int
    #it's not needed because we can simply use the freq_table to count and gain the frequency
    
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
        pairs = countAdjacent(freq_table)
        if pairs == {}:
            break # stop the train loop when empty dict(pairs) 

        pair = max(pairs,key=lambda k: (pairs[k], vocab[k[0]],vocab[k[1]]))# comparing the bytes rather than the token id
        # return the most frequent one and break the tie by loxic order by lambda method

        new_index = v+i
        freq_table = merge(freq_table,pair,new_index)
        # gain the freq_table by the merge function

        merges.append((vocab[pair[0]],vocab[pair[1]]))
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        # update the bpeParameters
        # pair: tuple(int,int). need to fetch the bytes for the merges

    return vocab,merges


def countAdjacent(freq_table:dict[tuple[int,...],int])->dict[tuple[int,int],int]:
    i = 0
    pairs = defaultdict(int)
    for key in freq_table:
        if len(key) > 1:
            for i in range(len(key)-1):
                # can only be used to 
                pairs[key[i],key[i+1]] += freq_table[key] 
        else:
            continue
        
    return dict(pairs) # transform pairs from defaultdict to dict


