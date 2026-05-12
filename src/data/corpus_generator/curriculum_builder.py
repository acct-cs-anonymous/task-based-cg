"""Curriculum building logic for synthetic data generation."""
import numpy as np


class CurriculumBuilder:
    """Handles curriculum generation strategies."""
    
    def __init__(self, functions_info, seq_len, cfg, token_manager):
        self.functions_info = functions_info
        self.seq_len = seq_len
        self.cfg = cfg
        self.token_manager = token_manager
        
    def get_function_list_for_curriculum_fixed_prompt(self, function_list):
        """Generate curriculum with fixed prompt length using identity padding."""
        function_lists = []
        max_combination_id = max(list(self.functions_info.values()))
        count = max_combination_id + 1
        
        for i in range(len(function_list) - 1):
            function_list_i = []
            # Add at least 2 functions from the function list
            for j in range(i + 2):
                function_list_i.append(function_list[j])
            
            # Add identity functions for remaining slots
            N = len(function_list)
            for j in range(N - i - 2):
                function_list_i.append("identity")
            
            sorted_function_list_i = tuple(function_list_i)
            
            if sorted_function_list_i not in self.functions_info:
                self.functions_info[sorted_function_list_i] = count
                count += 1
            function_lists.append(function_list_i)
            
        return function_lists
    
    def get_function_list_for_curriculum_variable_prompt(self, function_list):
        """Generate curriculum with variable prompt length."""
        orig_function_list = function_list
        function_lists = [orig_function_list]
        
        if len(orig_function_list) > 2:
            max_combination_id = max(list(self.functions_info.values()))
            count = max_combination_id + 1
            
            for i in range(2, len(function_list)):
                function_list_i = []
                for j in range(i + 1):
                    function_list_i.append(function_list[j])
                    
                sorted_function_list_i = tuple(function_list_i)
                
                if sorted_function_list_i not in self.functions_info:
                    self.functions_info[sorted_function_list_i] = count
                    count += 1
                function_lists.append(function_list_i)
                
        return function_lists
    
    def generate_curriculum_data(self, function_list, xstr1, xstr2, 
                                 filter_func, offset, function_dict, 
                                 apply_function_composition_fn):
        """Generate curriculum documents with fixed prompt length."""
        curriculum_function_list = []
        output_list = []
        curriculum_documents = []
        
        xstr1_tokens = np.array([self.token_manager.token_idx[c] for c in xstr1])
        xstr2_tokens = np.array([self.token_manager.token_idx[c] for c in xstr2])
        
        function_lists = self.get_function_list_for_curriculum_fixed_prompt(function_list)
        
        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            
            # Apply function composition
            outputs = apply_function_composition_fn(
                function_list_copy,
                function_dict,
                xstr1,
                xstr2,
                filter_func,
                offset,
                self.cfg.n_alphabets,
            )
            
            # Add padding to outputs
            outputs = self._pad_outputs(outputs)
            output_list.append(outputs)
        
        # Build curriculum documents
        for i in range(len(curriculum_function_list)):
            curriculum_doc = self._build_curriculum_document(
                curriculum_function_list[i],
                output_list[i],
                xstr1_tokens,
                xstr2_tokens
            )
            curriculum_documents.append(curriculum_doc)
            
        return curriculum_documents
    
    def generate_curriculum_data_variable_prompt(self, function_list, xstr1, xstr2,
                                                 filter_func, offset, function_dict,
                                                 apply_function_composition_fn):
        """Generate curriculum documents with variable prompt length."""
        curriculum_function_list = []
        output_list = []
        curriculum_documents = []
        
        xstr1_tokens = np.array([self.token_manager.token_idx[c] for c in xstr1])
        xstr2_tokens = np.array([self.token_manager.token_idx[c] for c in xstr2])
        
        function_lists = self.get_function_list_for_curriculum_variable_prompt(function_list)
        
        for function_list_copy in function_lists:
            curriculum_function_list.append(function_list_copy)
            
            # Apply function composition
            outputs = apply_function_composition_fn(
                function_list_copy,
                function_dict,
                xstr1,
                xstr2,
                filter_func,
                offset,
                self.cfg.n_alphabets
            )
            output_list.append(outputs)
        
        # Build curriculum documents
        for i in range(len(curriculum_function_list)):
            curriculum_doc = self._build_curriculum_document(
                curriculum_function_list[i],
                output_list[i],
                xstr1_tokens,
                xstr2_tokens
            )
            curriculum_documents.append(curriculum_doc)
            
        return curriculum_documents
    
    def _pad_outputs(self, outputs):
        """Add padding to outputs if necessary."""
        pad_length = 2 * self.seq_len
        if self.cfg.function.split.strategy in ["sort_map", "sort"]:
            pad_length = self.seq_len
            
        for i in range(len(outputs)):
            if self.cfg.prompt_length == "fixed":
                if len(outputs[i]) < pad_length:
                    outputs[i] = outputs[i] + self.token_manager.space_token * (
                        pad_length - len(outputs[i])
                    )
                else:
                    outputs[i] = outputs[i][:pad_length]
                    
        return outputs
    
    def _build_curriculum_document(self, function_list, outputs, 
                                   xstr1_tokens, xstr2_tokens):
        """Build a single curriculum document from components."""
        task_indices = self.token_manager.get_task_tokens(function_list)
        output_tokens = self.token_manager.get_output_tokens(outputs)
        
        curriculum_document = np.concatenate([
            self.token_manager.start_idx,
            task_indices,
            self.token_manager.sep_idx,
            xstr1_tokens,
            self.token_manager.sep_idx,
            xstr2_tokens,
            self.token_manager.sep_idx,
            output_tokens[-1],
            self.token_manager.end_idx,
        ])
        
        return curriculum_document