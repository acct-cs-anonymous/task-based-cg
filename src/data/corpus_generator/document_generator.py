"""Document generation for different prompt types."""
import numpy as np
from tqdm import tqdm
from .constants import VARIABLE_MAX_PROMPT_LENGTHS

class DocumentGenerator:
    """Handles generation of direct, step-by-step, and curriculum documents."""
    
    def __init__(self, cfg, token_manager, compositions_generator):
        self.cfg = cfg
        self.token_manager = token_manager
        self.compositions_generator = compositions_generator
        
        
    def sample_string(self):
        """Sample a random string of configured length."""
        alph = [chr(i + 97) for i in range(self.cfg.n_alphabets)]
        tokens = np.random.choice(
            alph, size=self.cfg.seq_len, replace=self.cfg.with_replacement
        )
        token_idx = [self.token_manager.token_idx[c] for c in tokens]
        tokens = "".join(tokens)
        return tokens, token_idx
    
    def generate_different_prompts_data(self, function_list):
        """Generate direct, step-by-step, and curriculum documents for a function list."""
        xstr1, xstr1_tokens = self.sample_string()
        xstr2, xstr2_tokens = self.sample_string()

        # Apply function composition
        outputs = self.compositions_generator.apply_function_composition(
            function_list,
            xstr1,
            xstr2
        )
        
        # Pad outputs
        outputs = self._pad_outputs(outputs)
        
        # Get tokens
        task_indices = self.token_manager.get_task_token_indices(function_list)
        output_tokens = self.token_manager.get_output_token_indices(outputs)
        
        # Build documents
        direct_document = self._build_direct_document(
            task_indices, xstr1_tokens, xstr2_tokens, output_tokens
        )
        
        step_by_step_document = self._build_step_by_step_document(
            task_indices, xstr1_tokens, xstr2_tokens, output_tokens
        )
        
            
        return direct_document, step_by_step_document
    
    def _pad_outputs(self, outputs):
        """Add padding to outputs based on strategy."""
        if self.cfg.function_type in ["diverse", "diverse2"]:
            pad_length = 2 * self.cfg.seq_len
        else:
            pad_length = self.cfg.seq_len
            
        for i in range(len(outputs)):
            if self.cfg.prompt_length == "fixed":
                if len(outputs[i]) < pad_length:
                    outputs[i] = outputs[i] + self.token_manager.space_token * (
                        pad_length - len(outputs[i])
                    )
                else:
                    outputs[i] = outputs[i][:pad_length]
                    
        return outputs
    
    def _build_direct_document(self, task_indices, xstr1_tokens, 
                               xstr2_tokens, output_tokens):
        """Build a direct document (input -> output)."""
        if self.cfg.function_type in ["diverse", "diverse2"]:
            return np.concatenate([
                np.array([self.token_manager.start_idx]),
                task_indices,
                np.array([self.token_manager.sep_idx]),
                xstr1_tokens,
                np.array([self.token_manager.sep_idx]),
                xstr2_tokens,
                np.array([self.token_manager.sep_idx]),
                output_tokens[-1],
                np.array([self.token_manager.end_idx]),
            ])
        else:
            return np.concatenate([
                np.array([self.token_manager.start_idx]),
                task_indices,
                np.array([self.token_manager.sep_idx]),
                xstr1_tokens,
                np.array([self.token_manager.sep_idx]),
                output_tokens[-1],
                np.array([self.token_manager.end_idx]),
            ])

    def _build_step_by_step_document(self, task_indices, xstr1_tokens,
                                     xstr2_tokens, output_tokens):
        """Build a step-by-step document (shows intermediate steps)."""
        
        if self.cfg.function_type in ["diverse", "diverse2"]:
            step_by_step_document = np.concatenate([
                np.array([self.token_manager.start_idx]),
                task_indices,
                np.array([self.token_manager.sep_idx]),
                xstr1_tokens,
                np.array([self.token_manager.sep_idx]),
                xstr2_tokens,
            ])
        else:
            step_by_step_document = np.concatenate([
                np.array([self.token_manager.start_idx]),
                task_indices,
                np.array([self.token_manager.sep_idx]),
                xstr1_tokens,
            ])
        
        for i in range(len(output_tokens)):
            step_by_step_document = np.concatenate([
                step_by_step_document,
                np.array([self.token_manager.sep_idx]),
                output_tokens[i]
            ])
        
        step_by_step_document = np.concatenate([
            step_by_step_document,
            np.array([self.token_manager.end_idx]),
        ])
        
        return step_by_step_document
    
    def generate_document(self, split, train_functions, test_functions):
        """Generate documents for a specific split (train/test/train_heldout)."""
        for function_list in train_functions:
            print(function_list)
            
        for function_list in test_functions:
            print(function_list)
            
        direct_documents = []
        step_by_step_documents = []
        
        # Determine functions and sample count
        functions, nsamples = self._get_split_parameters(
            split, train_functions, test_functions
        )
        
        # Generate documents
        for function_list in tqdm(functions):
            for i in tqdm(range(nsamples)):
                direct_doc, step_doc = (
                    self.generate_different_prompts_data(function_list)
                )
                direct_documents.append(direct_doc)
                step_by_step_documents.append(step_doc) 
        
        # Add padding
        return self.add_padding(
            direct_documents, step_by_step_documents
        )
    
    def _get_split_parameters(self, split, train_functions, test_functions):
        """Get functions and sample count for a split."""
        if split == "train":
            functions = train_functions
            if self.cfg.sample_efficiency_experiment:
                nsamples = self.cfg.nsamples
            else:
                ndocuments = self.cfg.ndocuments
                nsamples = int(ndocuments / len(train_functions))
        elif split == "train_heldout":
            functions = train_functions
            ndocuments = self.cfg.neval_documents
            nsamples = max(1, int(ndocuments / len(train_functions)))
        elif split == "test":
            functions = test_functions
            ndocuments = self.cfg.neval_documents
            nsamples = max(1, int(ndocuments / len(test_functions)))
        else:
            raise ValueError("Invalid split: {}".format(split))
            
        return functions, nsamples
    
    def add_padding(self, direct_documents, step_by_step_documents):
        """Add padding to documents for uniform length."""
        direct_documents = self._pad_documents(direct_documents, "direct")
        step_by_step_documents = self._pad_documents(step_by_step_documents, "step_by_step")
        
        # Log max lengths
        print("Max length of direct documents: ", 
              max(len(doc) for doc in direct_documents) if direct_documents else 0)
        print("Max length of step by step documents: ",
              max(len(doc) for doc in step_by_step_documents) if step_by_step_documents else 0)
        return direct_documents, step_by_step_documents
    
    def _pad_documents(self, documents, prompt_mode):
        """Pad a list of documents to uniform length."""
        if not documents:
            return documents
            
        if self.cfg.prompt_length == "variable":
            max_len = VARIABLE_MAX_PROMPT_LENGTHS[prompt_mode]
        else:
            max_len = max(len(doc) for doc in documents)
            
        return [
            (np.pad(doc, (0, max_len - len(doc)), 
                   constant_values=self.token_manager.space_idx)
             if len(doc) < max_len else doc)
            for doc in documents
        ]