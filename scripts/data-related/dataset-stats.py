#!/usr/bin/env python

"""dataset-stats.py

Usage:
  dataset-stats.py [--processed] <src>

Arguments:
  <src>             Path to file (segmented) file
"""

from docopt import docopt
import numpy as np

if __name__ == "__main__":
    arguments = docopt(__doc__)
    
    is_processed = bool(arguments["--processed"])

    if is_processed:
        print("Counting processed data")

    num_words = 0
    num_chars = 0
    src = arguments["<src>"]
    with open(src, "r") as fh:
        for line in fh:
            if is_processed:
                label = np.array(list(line.split("::")[0])).astype(int)
                num_words += np.sum(label)
                num_chars += label.shape[0]
            else:
                num_words += len(line.split("|"))
                num_chars += len(line.replace("|", ""))
    print(f"Stats: {src}")
    print(f"Num chars: {num_chars:,d}")
    print(f"Num words: {num_words:,d}")