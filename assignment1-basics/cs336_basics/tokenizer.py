import regex as re
from collections.abc import Iterable, Iterator
import pickle

#pretokenize the text by 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize_encode(
                    text:str,
                    special_tokens:list[str]
                    )->list[str]:
    sequences = []

    special_pat = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))

    if special_pat == "":# do not spilt the text by ""
         parts = [text] # tranform str to list[str]
         
    else:
         parts = re.split(f"({special_pat})",text)# f "({})" keep teh special tokens

    for part in parts: # part:str
        if part == "":
            continue
        elif part in special_tokens:
            sequences.append(part)
        else: # only deal with the normal text
            for match in re.finditer(PAT,part):
                pretoken = match.group() #pretoken:str
                sequences.append(pretoken)

    return sequences

#class Tokenizer store the BPE arguments trained, encode/decode use these arguments and it's more convenient to put them into the class

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int,bytes],
            merges: list[tuple[bytes,bytes]],
            special_tokens: list[str] | None = None
            ): # it can be list or None and if not used ,then None
            self.vocab = dict(vocab) # copy a new dict because self.dict should be updated with special_tokens and dont want to change the argument vocab
            self.merges = merges
            self.special_tokens = [] if special_tokens is None else special_tokens
            self.bytes_to_id = {token_bytes: token_id for token_id,token_bytes in vocab.items()} 
            self.merge_rank = {pair:rank for rank,pair in enumerate(merges)}# can be used to do dict lookup in O(1)

            # update self.vocab and self.bytes_to_id with the special_tokens
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8") # str to bytes

                if token_bytes not in self.bytes_to_id:
                    new_index = len(self.vocab)
                    self.vocab[new_index]= token_bytes
                    self.bytes_to_id[token_bytes] = new_index 
                     
                 

    @classmethod # make the method belong to the Class and use cls to function as the object tokenizer
    def from_files(
            cls,# can be used to call the attribute and function of the class Tokenizer
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
            ):
            
        
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens)

    def encode(self,text:str)->list[int]:
        # in each sequence earned from pretokenize, merge them first by the earned merges
        # after the merge operation ,look up the result in the vocab and turn it into the corresponding integer
        # which means a new dict bytes_to_id set up in __init__  can save the time of looking up the bytes

        # because the merge should also follow the order in merges, merge_rank is also need
        res = []

        pretokenized_text = pretokenize_encode(text,self.special_tokens)# gain the pretokenized text:list[str] with special tokens
        

        for token in pretokenized_text:

            if token not in self.special_tokens:# faster than iterate in vocab
                # merge in the token by looking up the merge_rank
                token2 = [bytes([k]) for k in token.encode("utf-8")] # str to bytes list because tuple cannot be modified
                
                while True:
                    best_pair = None
                    best_rank = float("inf")
                    i = 0

                    for pair in zip(token2,token2[1:]):# find the smallest rank pair in token2 to merge
                        if pair in self.merge_rank and self.merge_rank[pair]< best_rank:
                            best_rank = self.merge_rank[pair]
                            best_pair = pair
                    
                    if best_pair == None:
                        break

                    new_token = []
                    while i in range(len(token2)):
                        if i+1 in range(len(token2)) and (token2[i],token2[i+1]) == best_pair:
                            new_token.append(token2[i]+token2[i+1])
                            i+=2
                        else:
                            new_token.append(token2[i])
                            i+=1

                    token2 = new_token
                

                for tup in token2:# bytes of list(bytes,...)
                    res.append(self.bytes_to_id[tup])#bytes to token_id
                

            else:#convert the special tokens to token_id by self.vocab
                res.append(self.bytes_to_id[token.encode("utf-8")])# token:str -> bytes and bytes to id

        return res

    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id# use yield to return Iterator[int]

    def decode(self, ids:list[int]) -> str:
        byte_string = b"".join(self.vocab[i] for i in ids)
        text = byte_string.decode("utf-8",errors='replace')
        return text
