from gnninterpreter import *

import torch
from tqdm.auto import trange


# mutag = MUTAGDataset(seed=None)
# mutag_train, mutag_val = mutag.train_test_split(k=10)
# mutag_model = GCNClassifier(node_features=len(mutag.NODE_CLS),
#                             num_classes=len(mutag.GRAPH_CLS),
#                             hidden_channels=64,
#                             num_layers=3)

# print("Dataset : MUTAG")
# print("train base model: train size = ", len(mutag_train), ", test size = ", len(mutag_val))

# for epoch in trange(128):
#     print('#############################################')
#     print(f'Epoch: {epoch:03d}\n')
#     train_loss = mutag_train.fit_model(mutag_model, lr=0.001)
#     train_f1 = mutag_train.evaluate_model(mutag_model)
#     val_f1 = mutag_val.evaluate_model(mutag_model)
#     total_val, correct_val = mutag_val.eval_model_accuracy(mutag_model)
#     print('\n')
#     print(f'Epoch: {epoch:03d}, '
#           f'Train Loss: {train_loss:.4f}, '
#           f'Train F1: {train_f1}, '
#           f'Test F1: {val_f1}', 
#           f'Test Accuracy: {correct_val/total_val}')
    
# # print final test accuracy
# total_val, correct_val = mutag_val.eval_model_accuracy(mutag_model)
# print(f'Final Test Accuracy: {correct_val/total_val}')

# torch.save(mutag_model.state_dict(), 'ckpts/mutag.pt')


# MUTAGENICITY

mutagen = MutagenicityDataset(seed=None)
mutagen_train, mutagen_val = mutagen.train_test_split(k=10)
mutagen_model = GCNClassifier(node_features=len(mutagen.NODE_CLS),
                            num_classes=len(mutagen.GRAPH_CLS),
                            hidden_channels=64,
                            num_layers=3)

print("Dataset : Mutagenicity")
print("train base model: train size = ", len(mutagen_train), ", test size = ", len(mutagen_val))

for epoch in trange(128):
    print('#############################################')
    print(f'Epoch: {epoch:03d}\n')
    train_loss = mutagen_train.fit_model(mutagen_model, lr=0.001)
    train_f1 = mutagen_train.evaluate_model(mutagen_model)
    val_f1 = mutagen_val.evaluate_model(mutagen_model)
    total_val, correct_val = mutagen_val.eval_model_accuracy(mutagen_model)
    print('\n')
    print(f'Epoch: {epoch:03d}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Train F1: {train_f1}, '
          f'Test F1: {val_f1}', 
          f'Test Accuracy: {correct_val/total_val}')
    
# print final test accuracy
total_val, correct_val = mutagen_val.eval_model_accuracy(mutagen_model)
print(f'Final Test Accuracy: {correct_val/total_val}')

torch.save(mutagen_model.state_dict(), 'ckpts/mutagen_hd_64_epochs_128.pt')