# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'GMF'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre', 'NCF_Att', 'NCF_Att-pre']

# user_item_info: Use these info while training and testing
user_item_info = False
pretrain = False  # used only when "user_item_info" is activated.
# time_sort: If activated we will sort the data based on the time-stamp
# otherwise will be distributed to make sure that all the users are represented in each time-interval.
time_sorting = False
# rating_th: Filter data based on sorting. for example remove items that got rating less than 3
# to consider positive rating starting from 4 only. If set to zero that means no filtration will occur.
rating_th = 5
# window_split: If set, the data will be split into windows. Otherwise if zero the all data will be used
# Please note that this number in thousands.
window_split = 100


# Training
modes = ["training", "testing"]
mode = modes[0]

# paths
main_path = 'Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = 'weights/'
GMF_model_path = model_path + 'GMF_ratTh=4.pth'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP_org.pth'
NeuMF_model_path = model_path + 'NeuMF-pre_org.pth'
