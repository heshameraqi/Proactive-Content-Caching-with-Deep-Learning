from data import MovieLensData
import numpy as np

# Configurations
test_ratio = 0.1
batch_size = 256
split_mode = 'seq-aware'  # seq-aware or random
feedback = 'explicit'  # explicit or implicit

# Load data and print statistics
data = MovieLensData()
data.print_statistics()

# Afterwards, we put the above steps together and it will be used in the next section. The results are wrapped with Dataset and DataLoader.
# Note that the last_batch of DataLoader for training data is set to the rollover mode (The remaining samples are rolled over to the next epoch.) and orders are shuffled.
(train_u, train_i, train_r, train_inter), (test_u, test_i, test_r, test_inter) = data.split_load_data(split_mode, test_ratio, feedback)

train_set = gluon.data.ArrayDataset(np.array(train_u), np.array(train_i), np.array(train_r))
test_set = gluon.data.ArrayDataset(np.array(test_u), np.array(test_i), np.array(test_r))

train_iter = gluon.data.DataLoader(train_set, shuffle=True, last_batch='rollover', batch_size=batch_size)
test_iter = gluon.data.DataLoader(test_set, batch_size=batch_size)