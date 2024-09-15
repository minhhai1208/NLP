import pandas as pd
import numpy as np
from io import StringIO

data = """Get 3 Free Gift Cards (Total Value: $18) .

During the promotion, purchase any eligible product and receive three 1-month gift cards with the same benefits for free. You can share them with colleagues or classmates to experience WPS Pro membership benefits together. """
data = np.array([data])


print(type(data))
print(data.shape)