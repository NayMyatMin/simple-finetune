# dataeval module initialization
# This makes the dataeval directory a Python package

# Import core datasets
from . import coqa
from . import triviaqa
from . import nq_open
from . import SQuAD

# Import optional datasets if available
try:
    from . import TruthfulQA
except ImportError:
    pass

# Version information
__version__ = '0.1.0'
