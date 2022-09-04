import numpy as np
import torch
import config


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def E2EMetric(model, test_loader, top_k):
	topKPerUser, NDCG = [], []

	for user, item, target in test_loader:
		label = target['label']
		user = user.cuda()
		item = item.cuda()
		if config.user_item_info:
			user_info = target['user_info']
			item_info = target['item_info']
			user_info = user_info.float().cuda()
			item_info = item_info.float().cuda()
		else:
			user_info = None
			item_info = None
		predictions = model(user, item, user_info, item_info)
		_, indices = torch.topk(predictions, top_k)
		topKPerUser.append(indices)
	# Get top-k for the whole users:
	topKPerUser = torch.cat(topKPerUser)
	topKPerUser = topKPerUser.cpu().numpy()
	topItemID, countTopItems = np.unique(topKPerUser, return_counts=True)
	_, indices = torch.topk(torch.tensor(countTopItems), top_k)
	topKItem = topItemID[indices]

	# prepare GT:
	gtData = np.array(test_loader.dataset.features_ps)

	# Calculate Hit Rate:
	# ---------------------
	numHits = 0
	for item in gtData[:, 1]:
		if item in topKItem:
			numHits += 1
	acc = numHits / len(gtData[:, 1])

	"""
	# Calculate Hit Rate Old method (Bugy)
	uniValue, UniCount = np.unique(gtData[:, 1], return_counts=True)
	_, indices = torch.topk(torch.tensor(UniCount), top_k)
	gtTopKItems = uniValue[indices]

	# Calculate HitRate:
	numHits = sum(i in topKItem for i in gtTopKItems)
	acc = numHits/top_k
	"""

	return acc


def recE2Emetrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, target in test_loader:
		label = target['label']
		user = user.cuda()
		item = item.cuda()
		if config.user_item_info:
			user_info = target['user_info']
			item_info = target['item_info']
			user_info = user_info.float().cuda()
			item_info = item_info.float().cuda()
		else:
			user_info = None
			item_info = None
		predictions = model(user[0], item[0], user_info, item_info)
		predictions = predictions[item]  # filter predictions based on 100 item in the test data for fair comparison.
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, target in test_loader:
		label = target['label']
		user = user.cuda()
		item = item.cuda()
		if config.user_item_info:
			user_info = target['user_info']
			item_info = target['item_info']
			user_info = user_info.float().cuda()
			item_info = item_info.float().cuda()
		else:
			user_info = None
			item_info = None
		predictions = model(user, item, user_info, item_info)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)
