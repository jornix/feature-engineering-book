import pandas as pd
import numpy as np
import json
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# Using the previously loaded Yelp reviews DataFrame,
# compute the log transform of the Yelp review count.
# Note that we add 1 to the raw count to prevent the logarithm from
# exploding into negative infinity in case the count is zero.

biz_file = open("data/yelp_academic_dataset_business.json")
biz_df = pd.DataFrame([json.loads(x) for x in biz_file.readlines()])
biz_file.close()

biz_df["log_review_count"] = np.log10(biz_df["review_count"]+1)
