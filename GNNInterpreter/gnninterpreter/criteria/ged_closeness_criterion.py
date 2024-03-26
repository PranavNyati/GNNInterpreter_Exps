import torch
from torch import nn
import torch.nn.functional as F
from gnninterpreter.greed_model.neuro import models
import sys
sys.path.append('..')
sys.path.append('../..')


class GED_Closeness_Criterion(nn.Module):
    def __init__ (self, class_graphs, dataset_name='GED_Mutagenicity', discriminator=None, data_num_node_labels=14, k=10):
        super().__init__()
        self.class_graphs = class_graphs
        self.disriminator = discriminator # model to be explained
        self.ged_greed_model = models.NormGEDModel(8, data_num_node_labels, 64, 64)
        self.ged_greed_model.load_state_dict(torch.load(f'gnninterpreter/greed_model/runlogs/{dataset_name}/best_model.pt'))
        print("GED Greed Model loaded")
        self.k = k
        
        
    def find_closest_k_class_graphs_index(self, sampled_graph):
        # print("Finding closest k class graphs")
        indices = []
        # print(f"type of sampled graph: {type(sampled_graph[0])}")
        sampled_graph = self.remove_extra_graph_attributes(sampled_graph[0])
            

        # for i, class_graph in enumerate(self.class_graphs):
            
        #     class_graph = self.remove_extra_graph_attributes(class_graph)            
        #     ged = self.ged_greed_model.predict_inner([class_graph], [sampled_graph])
        #     # print(f"GED between class graph {i} and sampled graph is {ged[0]}")
        #     indices.append((ged[0], i))
            
        # parallelize the above code
        ged = self.ged_greed_model.predict_inner(self.class_graphs, [sampled_graph]*len(self.class_graphs))
        indices = [(ged[i], i) for i in range(len(ged))]
            
        # sort indices based on increasing order of ged usign lambda function
        indices.sort(key = lambda x: x[0])
        # now return the first k indices
        if (len(indices) < self.k):
            return [i[1] for i in indices]  
        
        top_k_indices = [i[1] for i in indices[:self.k]]
        return top_k_indices
        
    def forward(self, sampled_graph):
        print("GED Closeness Criterion forward pass called")
        top_k_indices = self.find_closest_k_class_graphs_index(sampled_graph)
        print(f"Top {self.k} class graphs indices: {top_k_indices}")
        # now get the GNN (to be explained) embeddings of the top k class graphs and the sampled graph
        top_k_embeddings = []
        sampled_graph_embedding = self.disriminator(sampled_graph)["embeds"]
        # print("Sampled Graph Embedding shape: ", sampled_graph_embedding.shape)
        # gcn_out = self.disriminator(self.class_graphs[top_k_indices])["embeds"]
        gcn_out = []
        for i in top_k_indices:
            class_graph = self.class_graphs[i]
            # class_graph = self.remove_extra_graph_attributes(class_graph)
            gcn_out.append(self.disriminator(class_graph)["embeds"])
        
        
        # now calculate the cosine similarity between the sampled graph and the top k class graphs 
        loss = 0.0
        for i in range(self.k):
            # print(f"K={i}:")
            # print(f"Cosine Sim: {F.cosine_similarity(sampled_graph_embedding, gcn_out[i], dim=1)}")
            loss += F.cosine_similarity(sampled_graph_embedding, gcn_out[i], dim=1).item()
            # print(f"Cos Sim b/w sampled graph and class graph {i} is {F.cosine_similarity(sampled_graph_embedding, gcn_out[i], dim=1):.4f}")
        
        loss = 100*loss    
        
        return (loss/self.k)
        
        
    def remove_extra_graph_attributes(self, graph):
        
        for k in list(graph.keys()):
            if k not in ['x', 'edge_index']:
                graph[k] = None
                
        return graph