"""Token management for synthetic data generation."""
import numpy as np
import logging
from src.data.corpus_generator.constants import SPECIAL_TOKENS
import os


class TokenManager:
    """Handles token initialization, encoding, and decoding."""
    
    def __init__(self, n_alphabets=None, function_dict=None, load_path=None):
        """
        Initialize TokenManager.
        
        Args:
            n_alphabets: Number of alphabet tokens (a-z). Required if load_path is None.
            function_dict: Dictionary of functions. Required if load_path is None.
            load_path: Path to directory containing token.pkl, token_idx.pkl, and functions_info.pkl.
                      If provided, tokens will be loaded from files instead of initialized.
        """
        self.special_tokens = SPECIAL_TOKENS
        self.token = {}
        self.token_idx = {}
        self.logger = logging.getLogger(__name__)
        
        # Special token attributes
        self.start_token = self.special_tokens["START"]
        self.space_token = self.special_tokens["SPACE"]
        self.sep_token = self.special_tokens["SEP"]
        self.null_token = self.special_tokens["NULL"]
        self.end_token = self.special_tokens["END"]
        
        if load_path is not None:
            self._load_from_path(load_path)
        else:
            if n_alphabets is None or function_dict is None:
                raise ValueError("n_alphabets and function_dict are required when load_path is not provided")
            self.n_alphabets = n_alphabets
            self.function_dict = function_dict
            self.init_tokens()
    
    def _load_from_path(self, load_path):
        """Load token dictionaries from pickle files."""
        token_fname = os.path.join(load_path, "token.pkl")
        token_idx_fname = os.path.join(load_path, "token_idx.pkl")
        functions_info_fname = os.path.join(load_path, "functions_info.pkl")
        
        self.token = np.load(token_fname, allow_pickle=True)
        self.token_idx = np.load(token_idx_fname, allow_pickle=True)
        self.functions_info = np.load(functions_info_fname, allow_pickle=True)
            
        # Create index arrays for quick access
        self._create_indices()
        
        self.logger.info("Tokens loaded from: {}".format(load_path))
        self.logger.info("Tokens: {}".format(self.token))
        self.logger.info("Token indices: {}".format(self.token_idx))

    def map_tokens(self, token_map, tokenizer):
        """Map tokens to new tokens."""
        for prev_token_idx, new_token_idx in token_map.items():
            # update token_idx
            new_token = tokenizer.convert_ids_to_tokens(new_token_idx)
            self.token_idx[new_token] = new_token_idx
            # update token
            self.token[new_token_idx] = new_token
            # remove prev_token_idx from token_idx
            del self.token[prev_token_idx]
        
    def init_tokens(self):
        """Initialize alphabet, special, and function tokens."""
        # Create alphabet tokens (a-z)
        for i in range(self.n_alphabets):
            self.token[i] = chr(i + 97)
            self.token_idx[chr(i + 97)] = i
        
        # Add special tokens
        sp_token_count = self._add_special_tokens()
        
        # Add function tokens
        self._add_function_tokens(sp_token_count)
        print(self.token)
        print(self.token_idx)
        
        # Create index arrays for quick access
        self._create_indices()
        
        self.logger.info("Tokens: {}".format(self.token))
        self.logger.info("Token indices: {}".format(self.token_idx))
        
    def _add_special_tokens(self):
        """Add special tokens."""
        sp_token_count = 0
        for token in self.special_tokens.values():
            self.token[self.n_alphabets + sp_token_count] = token
            self.token_idx[token] = self.n_alphabets + sp_token_count
            sp_token_count += 1
        return sp_token_count
    
    def _add_function_tokens(self, offset):
        """Add function tokens."""
        count = 0
        for token in self.function_dict.keys():
            self.token[self.n_alphabets + offset + count] = token
            self.token_idx[token] = self.n_alphabets + offset + count
            count += 1
    
    def _create_indices(self):
        """Create numpy arrays for frequently used token indices."""
        
        self.start_idx = self.token_idx[self.start_token]
        self.sep_idx = self.token_idx[self.sep_token]
        self.end_idx = self.token_idx[self.end_token]
        
        self.space_idx = self.token_idx[self.space_token]
        self.null_idx = self.token_idx[self.null_token]
    
    def decode(self, token_indices, return_list=False):
        """Decode token indices to human-readable string."""
        txt_list = []
        for i in token_indices:
            if i in self.token:
                if self.token[i] is None:
                    txt_list.append("<UNK>")
                    continue
                txt_list.append(self.token[i])
                if not return_list:
                    txt_list.append(" ")
        
        # Remove last space
        if txt_list and txt_list[-1] == " ":
            txt_list = txt_list[:-1]
            
        return "".join(txt_list) if not return_list else txt_list
    
    def encode(self, txt):
        """Encode string to token indices."""
        return [self.token_idx[c] for c in txt]
    
    def get_task_token_indices(self, function_list):
        """Convert function list to token indices."""
        return np.array([self.token_idx[fn_name] for fn_name in function_list])
    
    def get_output_token_indices(self, outputs):
        """Convert output strings to token indices."""
        output_indices = []
        for output in outputs:
            output_idx = []
            if len(output) == 0:
                output_idx.append(self.token_idx["<NULL>"])
            else:
                output_idx = [self.token_idx[output[i]] for i in range(len(output))]
            output_indices.append(output_idx)
        return output_indices
    
    def get_vocab_len(self):
        """Get vocabulary length."""
        return len(self.token_idx)
    
    def get_sep_pos(self, sample):
        """Get the position of the last separator token in the sample."""
        sep_pos = np.where(sample == self.sep_idx)[0][-1]
        return sep_pos

    def get_seq_info(self, sample, function_type):
        # Find last separator position
        sep_positions = np.where(sample == self.sep_idx)[0]
        last_sep_pos = sep_positions[-1] if len(sep_positions) > 0 else 0

        if function_type == "uniform":
            third_sep_pos = sep_positions[1]
        else:
            third_sep_pos = sep_positions[2]
        
        # Find end position
        end_positions = np.where(sample == self.end_idx)[0]
        end_pos = end_positions[0] if len(end_positions) > 0 else len(sample)
        
        # For direct/curriculum mode, prompt ends at last separator
        # For step_by_step, prompt ends after first function call
        prompt_pos_end = third_sep_pos + 1
        
        # Calculate how many tokens to generate
        extra_space_tokens = len(sample) - end_pos - 1
        new_len = len(sample) - prompt_pos_end - extra_space_tokens
        
        input_data_start = int(sep_positions[0]) + 1
        input_data_end = int(last_sep_pos)

        return {
            "prompt_pos_end": prompt_pos_end,
            "last_sep_pos": last_sep_pos,
            "end_pos": end_pos,
            "new_len": new_len,
            "input_data_start": input_data_start,
            "input_data_end": input_data_end,
        }

    def get_input_string(self, doc, function_type):
        if function_type == "diverse":
            third_sep_pos = np.where(doc == self.sep_idx)[0][2]
        else:
            third_sep_pos = np.where(doc == self.sep_idx)[0][1]
        first_sep_pos = np.where(doc == self.sep_idx)[0][0]
        input_string = doc[first_sep_pos + 1 : third_sep_pos]
        return input_string


    def get_output_string(self, doc, function_type, token_map=None):
        
        if function_type == "diverse":
            third_sep_pos = np.where(doc == self.sep_idx)[0][2]
        else:
            third_sep_pos = np.where(doc == self.sep_idx)[0][1]
        end_token_pos = np.where(doc == self.end_idx)[0][0]
        output_string = doc[third_sep_pos + 1 : end_token_pos]
        return output_string

    def get_function_list(self, doc, token_map=None):
        """Get the function list from the document."""
        # get first separator position
        # print("Token Dictionary: ", self.token)
        # print("Token Index Dictionary: ", self.token_idx)
        # print("Token Map: ", token_map)
        # print(doc, self.sep_idx)
        first_sep_pos = np.where(doc == self.sep_idx)[0][0]
        function_list = doc[1:first_sep_pos]
        return function_list


