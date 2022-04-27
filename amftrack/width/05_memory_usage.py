import sys


def function(n):
    # d = {f"{i}": {} for i in range(n)}
    d = [set() for i in range(n)]
    # d = [i for i in range(n)]
    print(f"Memory usage: {sys.getsizeof(d)/1000000000} Go")
    return d


if __name__ == "__main__":
    function(1000_000)
