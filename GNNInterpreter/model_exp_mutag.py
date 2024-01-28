from gnninterpreter import *
import torch

dataset = MUTAGDataset(seed=12345)

model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                      num_classes=len(dataset.GRAPH_CLS),
                      hidden_channels=64,
                      num_layers=3)

model.load_state_dict(torch.load('ckpts/mutag.pt'))

eval= dataset.evaluate_model(model)
print("Evaluation results: ", eval)

mean_embeds = dataset.mean_embeddings(model)

trainer = {}
sampler = {}

# MUTAGEN
# print("\nMutagenic Class (Class 1)\n")

# cls_idx = 1
# trainer[cls_idx] = Trainer(
#     sampler=(s := GraphSampler(
#         max_nodes=20,
#         num_node_cls=len(dataset.NODE_CLS),
#         temperature=0.15,
#         learn_node_feat=True
#     )),
#     discriminator=model,
#     criterion=WeightedCriterion([
#         dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
#         dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=10),
#         dict(key="logits", criterion=MeanPenalty(), weight=0),
#         dict(key="omega", criterion=NormPenalty(order=1), weight=1),
#         dict(key="omega", criterion=NormPenalty(order=2), weight=1),
#         dict(key="xi", criterion=NormPenalty(order=1), weight=0),
#         dict(key="xi", criterion=NormPenalty(order=2), weight=0),
#         # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
#         # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
#         dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
#     ]),
#     optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
#     scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
#     dataset=dataset,
#     budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
#     target_probs={cls_idx: (0.9, 1)},
#     k_samples=16
# )

# trainer[1].train(2000)

# g_mutag = trainer[1].evaluate(threshold=0.5, show=True, path='results/mutag_factual_exp/mutag_class_GCN.png')

# trainer[1].save(g_mutag, cls_idx)

print("\nNon-Mutagenic Class (Class 0)\n")

## NON-MUTAGEN
cls_idx = 0
trainer[cls_idx] = Trainer(
    sampler=(s := GraphSampler(
        max_nodes=20,
        num_node_cls=len(dataset.NODE_CLS),
        temperature=0.15,
        learn_node_feat=True
    )),
    discriminator=model,
    criterion=WeightedCriterion([
        dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=1),
        dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=10),
        dict(key="logits", criterion=MeanPenalty(), weight=0),
        dict(key="omega", criterion=NormPenalty(order=1), weight=1),
        dict(key="omega", criterion=NormPenalty(order=2), weight=1),
        dict(key="xi", criterion=NormPenalty(order=1), weight=0),
        dict(key="xi", criterion=NormPenalty(order=2), weight=0),
        # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
        # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
        dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
    ]),
    optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
    scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
    dataset=dataset,
    budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
    target_probs={cls_idx: (0.9, 1)},
    k_samples=16
)

trainer[0].train(2000)

g_non_mutag = trainer[0].evaluate(threshold=0.5, show=True, path='results/mutag_factual_exp/non_mutag_class_GCN.png')
trainer[0].save(g_non_mutag, cls_idx)
