import pickle
import pandas as pd

cls = pd.StringDtype
old = cls.__init__

def compat(self, *args, **kwargs):
    print("STRING ARGS", args, kwargs)
    return old(self)

cls.__init__ = compat
frame = pickle.load(open("model/artifacts/df.pkl", "rb"))
print(frame.shape, frame.dtypes.head())
