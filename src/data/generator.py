"""Main synthetic data generator class - refactored and modularized."""
from init import read_config, ROOT_DIR
from src.data.composition_generator.compositions import CompositionsGenerator
from src.data.corpus_generator.token_manager import TokenManager
from src.data.corpus_generator.document_generator import DocumentGenerator
from src.data.corpus_generator.data_storage import DataStorage



class SyntheticDataGenerator:
    """
    Generates synthetic composition data of the form:
    f1, f2, ..., fn, x, f1(x), f2(x), ..., fn(x)
    
    This class orchestrates the data generation process by coordinating
    specialized modules for token management,
    document generation, and data storage.
    """

    def __init__(self, cfg, logger):
        """
        Initialize the synthetic data generator.
        
        Args:
            cfg: Configuration object
            train_functions: List of function compositions for training
            test_functions: List of function compositions for testing
            function_dict: Dictionary mapping function names to implementations
            functions_info: Dictionary tracking function combinations
            apply_function_composition_fn: Function to apply compositions
        """
        self.cfg = cfg
        self.compositions_generator = CompositionsGenerator(cfg)
        self.train_functions, self.test_functions, self.functions_info = self.compositions_generator.get_train_test_compositions()
        # Initialize specialized modules
        self.storage = DataStorage(cfg, ROOT_DIR)
        self.logger = logger
        self.logger.info(f"Train functions: {len(self.train_functions)}")
        self.logger.info(f"Test functions: {len(self.test_functions)}")
        
        self.token_manager = TokenManager(
            cfg.n_alphabets,
            self.compositions_generator.function_dict
        )
        self.token_manager.init_tokens()
        self.document_generator = DocumentGenerator(
            cfg,
            self.token_manager,
            self.compositions_generator
        )

    def generate_corpus(self):
        """Generate the complete corpus (train, test, train_heldout)."""
        self.logger.info("Starting corpus generation...")
        
        # Generate documents for all splits
        train_direct, train_step = (
            self.document_generator.generate_document(
                "train", self.train_functions, self.test_functions
            )
        )
        
        train_heldout_direct, train_heldout_step = (
            self.document_generator.generate_document(
                "train_heldout", self.train_functions, self.test_functions
            )
        )
        
        test_direct, test_step = (
            self.document_generator.generate_document(
                "test", self.train_functions, self.test_functions
            )
        )
        
        # Log samples
        self._log_samples(train_direct, train_step)
        
        # Store corpus
        self.corpus = {
            "train_direct": train_direct,
            "train_step_by_step": train_step,
            "test_direct": test_direct,
            "test_step_by_step": test_step,
            "train_heldout_direct": train_heldout_direct,
            "train_heldout_step_by_step": train_heldout_step,
        }
        
        self.logger.info("Corpus generation complete!")

    def _log_samples(self, train_direct, train_step):
        """Log sample documents for inspection."""
        # Log direct documents
        for i in range(min(1, len(train_direct))):
            self.logger.info("Direct documents")
            self.logger.info(len(train_direct[i]))
            self.logger.info(train_direct[i])
            self.logger.info(self.token_manager.decode(train_direct[i]))
        
        # Log step-by-step documents
        for i in range(min(1, len(train_step))):
            self.logger.info(len(train_step[i]))
            self.logger.info(train_step[i])
            self.logger.info("Step by step documents")
            self.logger.info(self.token_manager.decode(train_step[i]))
        

    def store_data(self):
        """Store the generated corpus to disk."""
        self.storage.store_data(
            self.corpus,
            self.token_manager.token,
            self.token_manager.token_idx,
            self.functions_info
        )


def main():
    """Main entry point for data generation."""
    # Read config
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    cfg = read_config(cfg_path)
    
    # Create synthetic data generator
    synthetic_data_generator = SyntheticDataGenerator(cfg)
    synthetic_data_generator.generate_corpus()
    synthetic_data_generator.store_data()


if __name__ == "__main__":
    main()