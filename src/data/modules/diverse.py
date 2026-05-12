import json
from init import ROOT_DIR
from src.data.modules.randomFunctions import MapRandom
class DiverseFunctionsV1:
    @staticmethod
    def identity(xstr):
        return xstr

    @staticmethod
    def map(xstr, offset=1, n_alphabets=26):
        """ Shifts the characters in the string by the offset."""
        return "".join(
            chr((ord(c) - ord("a") + offset) % n_alphabets + ord("a")) for c in xstr
        )

    @staticmethod
    def sort(xstr):
        """ Sorts the characters in the string."""
        return "".join(sorted(xstr))

    @staticmethod
    def join(xstr1, xstr2):
        """ Joins two strings together."""
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """ Returns the union of two strings."""
        ans = []
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def mode(xstr1):
        """
        Finds the letter with the maximum occurrences in the string.
        If multiple characters have the same occurrence, chooses the alphabetically smallest one.
        """
        if len(xstr1) == 0:
            return ""
        from collections import Counter

        # Count occurrences of each character
        char_count = Counter(xstr1)

        # Find the maximum occurrence count
        max_count = max(char_count.values())

        # Filter characters with the maximum count and return the smallest alphabetically
        max_chars = [char for char, count in char_count.items() if count == max_count]
        return min(max_chars)

    def filter(xstr, filter_func=lambda x: x in "aeiou"):
        """
        Filters a string based on a filter function.
        """
        # max length of the string is 12
        return "".join(filter(filter_func, xstr))


class DiverseFunctionsV2:
    @staticmethod
    def identity(xstr):
        return xstr

    @staticmethod
    def map(xstr):
        # use random mapping from RandomMapping class
        return MapRandom.map_random(seed=1)(xstr)


    @staticmethod
    def sort(xstr):
        """ Sorts the characters in the string."""
        return "".join(sorted(xstr))

    @staticmethod
    def join(xstr1, xstr2):
        """ Joins two strings together String1 + String2."""
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """ Returns the union of two strings Set(String1) U Set(String2)."""
        ans = []
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def reverse_sort(xstr1):
        """ Reverse sorts the characters in the string."""
        return "".join(sorted(xstr1, reverse=True))

    def filter_consonants(xstr):
        """ Filters a string based on a filter function."""
        return "".join(filter(lambda x: x not in "aeiou", xstr))


DIVERSE_FUNCTIONS = {
    "sort": DiverseFunctionsV1.sort,
    "join": DiverseFunctionsV1.join,
    "filter": DiverseFunctionsV1.filter,
    "map": DiverseFunctionsV1.map,
    "union": DiverseFunctionsV1.union,
    "mode": DiverseFunctionsV1.mode,
    "identity": DiverseFunctionsV1.identity,
}


DIVERSE_2_FUNCTIONS = {
    "sort": DiverseFunctionsV2.sort,
    "join": DiverseFunctionsV2.join,
    "map": DiverseFunctionsV2.map,
    "union": DiverseFunctionsV2.union,
    "reverse": DiverseFunctionsV2.reverse_sort,
    "filter": DiverseFunctionsV2.filter_consonants,
    "identity": DiverseFunctionsV2.identity,
}
