import torch

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """Helper method to run kNN predictions on features based on a feature bank, normalizing each feature to unit length."""

    # Normalize feature to unit length
    feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-6)

    # Normalize feature bank to unit length
    feature_bank = feature_bank / (feature_bank.norm(dim=1, keepdim=True) + 1e-6)

    # Compute cosine similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank.T)

    # Get top K neighbors; sim_weight: [B, K], sim_indices: [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # Retrieve labels of these top K neighbors; sim_labels: [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    # Reweight similarities
    sim_weight = (sim_weight / knn_t).exp()

    # Calculate weighted scores for each class; one_hot_label: [B*K, C]
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    # Weighted score; pred_scores: [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    # Sort the scores to get predicted labels
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    # If you need the top-1 label for each feature vector
    top_pred_labels = pred_labels[:, 0]

    return top_pred_labels