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


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	#for user, item, label, user_info, item_info in test_loader:
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
