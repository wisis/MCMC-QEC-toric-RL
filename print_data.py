from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import os
import sys
from src.util import MCMCDataReader

if __name__ == '__main__':
    file_path=os.path.join(sys.argv[1])
    
    iterator = MCMCDataReader(file_path, 5)
    while iterator.has_next():
        print('Datapoint nr: '+ str(iterator.current_index() + 1))
        print(iterator.next())
