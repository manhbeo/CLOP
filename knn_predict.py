import torch

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    '''
    Helper method to run k-Nearest Neighbors (kNN) predictions on features based on a feature bank.

    Parameters:
    - feature (torch.Tensor): The query features for which predictions are to be made. Shape [B, D] where B is the batch size, and D is the feature dimension.
    - feature_bank (torch.Tensor): The reference feature bank (stored features) to be compared against. Shape [N, D] where N is the number of stored features.
    - feature_labels (torch.Tensor): The labels corresponding to the stored features in the feature bank. Shape [N].
    - classes (int): The number of classes.
    - knn_k (int): The number of nearest neighbors to consider.
    - knn_t (float): A temperature parameter for scaling the similarity weights.

    Returns:
    - top_pred_labels (torch.Tensor): The predicted class labels for each feature vector in the query features. Shape [B].
    '''
    feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-6)
    feature_bank = feature_bank / (feature_bank.norm(dim=1, keepdim=True) + 1e-6)

    sim_matrix = torch.mm(feature, feature_bank.T)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    top_pred_labels = pred_labels[:, 0]

    return top_pred_labels