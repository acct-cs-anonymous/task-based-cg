import numpy as np
import random
import string
    
class MapRandom:
    @staticmethod
    def map_random(seed, n_alphabets=26):
        """
        Returns a function that maps a string using a random mapping generated with the given seed.
        Also provides a way to get the mapping dictionary for inspection/storage.
        """
        np.random.seed(seed)
        random.seed(seed)
        # mapping is without replacement
        # sample n_alphabets characters from a-z
        sampled_characters = random.sample(list(string.ascii_lowercase), n_alphabets)
        mapping = {chr(i + ord("a")): sampled_characters[i] for i in range(n_alphabets)}

        def map_func(xstr):
            return "".join(mapping[c] for c in xstr)

        map_func.mapping = mapping  # Attach mapping dict for external access
        
        return map_func