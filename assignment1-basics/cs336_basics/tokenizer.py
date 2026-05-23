import regex as re

#pretokenize the text by 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize_encode(
                    text:str,
                    special_tokens:list[str]
                    )->list[str]:
    sequences = []

    special_pat = "|".join(re.escape(tok) for tok in special_tokens)

    if special_pat == "":# do not spilt the text by ""
         parts = [text] # tranform str to list[str]
         
    else:
         parts = re.split(f"({special_pat})",text)# f "({})" keep teh special tokens

    for part in parts: # part:str
        if part == "" or part in special_tokens:
            continue
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
            self.merge_rank = {pair:rank for pair, rank in enumerate(merges)}# can be used to do dict lookup in O(1)

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
                token2 = tuple(bytes([k]) for k in token.encode("utf-8")) # str to bytes tuple
                
                i = 0

                while i in range(len(token)):# loop of merge inside the bytes tuple, avoid break the border
                    token3 = ()# store the merged bytes tuple
                    if i+1 in range(len(token)):
                        if (token2[i],token2[i+1]) in self.merge_rank:
                            token3 += (token2[i]+token2[i+1],)
                            i+=2
                        else:
                            token3 += (token[i],)
                    else:
                        token3 += (token[i],)
                
                for tup in token3:# bytes of tuple(bytes,...)
                    res.append(self.bytes_to_id[tup])#bytes to token_id
                

            else:#convert the special tokens to token_id by self.vocab
                res.append(self.bytes_to_id(token.encode("utf-8")))# token:str -> bytes and bytes to id

        return res

    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:

    def decode(self, ids:list[int]) -> str:
