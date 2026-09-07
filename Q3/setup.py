"""Part 0: Setup"""

import nltk
nltk.download('brown')
nltk.download('punkt')

import random
import re
import string
import time
import pickle
from collections import Counter, defaultdict

from nltk.corpus import brown

random.seed(42)
print("Setup complete.")
