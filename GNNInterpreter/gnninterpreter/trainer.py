import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import copy
import secrets
import os
import pickle
import glob
import torch.nn.functional as F
import torch_geometric as pyg
from typing import Type, TypeVar
# from graph_sampler import GraphSampler

# TODO: refactor
# from .datasets import *


class Trainer:
    def __init__(self,
                 sampler,  ### graph sampler object
                 discriminator, ### model to be explained
                 criterion, ### total weighted loss objective including the regularizers
                 scheduler, ### scheduler for learning rate decay
                 optimizer, ### optimizer for the sampler (SGD, Adam, etc.)
                 dataset, 
                 budget_penalty=None, ### expected maximum number of edges in the sampled graph
                 seed=None,
                 target_probs: dict[tuple[float, float]] = None,  ### gives the range of probabilities for target class to be explained
                 ### i.e., if target_probs = {1: (0.9, 1)}, then the target class is 1 and we want the prob of the generated explanation to be in the range [0.9, 1]
                 k_samples=32): ### ??
        self.k = k_samples
        self.target_probs = target_probs
        self.sampler = sampler
        self.discriminator = discriminator
        self.criterion = criterion
        self.budget_penalty = budget_penalty
        self.scheduler = scheduler
        self.optimizer = optimizer if isinstance(optimizer, list) else [optimizer]
        self.dataset = dataset
        self.iteration = 0
        self.seed = seed
        # set seed for networkx
        # if seed is not None:
        #     nx.set_random_seed(seed)

    def probe(self, cls=None, discrete=False):
        graph = self.sampler(k=self.k, discrete=discrete, seed=self.seed)
        logits = self.discriminator(graph, edge_weight=graph.edge_weight)["logits"].mean(dim=0).tolist()
        print(f"probe: {logits=}")
        if cls is not None:
            return logits[cls]
        return logits

    def detailed_probe(self):
            
        det_probe = pd.DataFrame(dict(
            logits_discrete=(ld := self.probe(discrete=True)),
            logits_continuous=(lc := self.probe(discrete=False)),
            prob_discrete=F.softmax(torch.tensor(ld), dim=0).tolist(),
            prob_continuous=F.softmax(torch.tensor(lc), dim=0).tolist(),
        ))
        print("detailed_probe: ", det_probe)

    def warmup(self, iterations, cls, score):
        orig_criterion = copy.deepcopy(self.criterion)
        orig_iteration = self.iteration
        
        ### Pranav : (What is this part doing?)
        while self.probe(cls, discrete=True) < score: 
            self.criterion = copy.deepcopy(orig_criterion)
            self.iteration = orig_iteration
            self.sampler.init(seed=self.seed)
            self.train(iterations)

    def train(self, iterations):
        self.bkup_state = copy.deepcopy(self.sampler.state_dict())
        self.bkup_criterion = copy.deepcopy(self.criterion)
        self.bkup_iteration = self.iteration
        self.discriminator.eval()
        self.sampler.train()
        budget_penalty_weight = 1
        
        # for each iteration:
        for _ in (bar := tqdm(range(int(iterations)), initial=self.iteration, total=self.iteration+iterations)):
            for opt in self.optimizer:
                opt.zero_grad()
            
            ### sample both continuous and discrete graphs 
            ### Pranav: Why is the k argument different for continuous and discrete graphs?
            cont_data = self.sampler(k=self.k, mode='continuous', seed=self.seed)
            disc_data = self.sampler(k=1, mode='discrete', seed=self.seed, expected=True)
            
            # TODO: potential bug  ### Pranav: What is the bug is the author referring to?
            
            # get the logits, probs, and embeddings for the continuous and discrete graphs
            cont_out = self.discriminator(cont_data, edge_weight=cont_data.edge_weight)
            disc_out = self.discriminator(disc_data, edge_weight=disc_data.edge_weight)
            
            # if there is some target probability and the probability of the target explanation is within the range of target_probs 
            # and the expected maximum number of edges in the sampled graph is less than the pre-defined budget, then current graph is a good explanation (break from the loop)
            if self.target_probs and all([
                min_p <= disc_out["probs"][0, classes].item() <= max_p
                for classes, (min_p, max_p) in self.target_probs.items()
            ]):
                print("Hello")
                print("\nPrediction of the sample: ", (disc_out["probs"][0, classes].item() for classes, (min_p, max_p) in self.target_probs.items()))
                if self.budget_penalty and self.sampler.expected_m <= self.budget_penalty.budget:
                    # print(f"\nPrediction of the sample: {disc_out['probs'][0, classes].item()}" for classes in self.target_probs)
                    print(f"Expected number of edges of sample: {self.sampler.expected_m}; Budget: {self.budget_penalty.budget}")
                    print("Current explanation has high prediction probability and low expected maximum number of edges. Hence, it is a good explanation.")
                    print("Breaking from train loop!\n")
                    break
                
                budget_penalty_weight *= 1.1 ### if the expected max no of edges > budget, then increase the budget penalty weight by 10%
            else:
                ### Doubt: Why decrease budget penalty if expln doesn't achieve target prob?
                ### How will budget penalty decrease help in achieving target prob?
                budget_penalty_weight *= 0.95 ### if the generated explanation doesn't achieve the target probability, then decrease the budget penalty weight by 5%

            loss = self.criterion(cont_out | self.sampler.to_dict()) ### Why is self.sample.to_dict() used here? 
            if self.budget_penalty:
                loss += self.budget_penalty(self.sampler.theta) * budget_penalty_weight
            loss.backward()  # Back-propagate gradients

            # print(self.sampler.omega.grad)

            for opt in self.optimizer:
                opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            size = self.sampler.expected_m
            scores = disc_out["logits"].mean(axis=0).tolist()
            score_dict = {v: scores[k] for k, v in self.dataset.GRAPH_CLS.items()}
            penalty_weight = {'bpw': budget_penalty_weight} if self.budget_penalty else {}
            bar.set_postfix({'size': size} | penalty_weight | score_dict)
            print(f"iteration={self.iteration}, loss={loss.item():.2f}, {size=}, scores={score_dict}")
            self.iteration += 1
        else:
            return False
        return True

    def undo(self):
        self.sampler.load_state_dict(self.bkup_state)
        self.criterion = copy.deepcopy(self.bkup_criterion)
        self.iteration = self.bkup_iteration

    @torch.no_grad()
    def predict(self, G):
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        return self.discriminator(batch)

    @torch.no_grad()
    def quantatitive(self, sample_size=1000, sample_fn=None):
        sample_fn = sample_fn or (lambda: self.evaluate(bernoulli=True))
        p = []
        for i in range(1000):
            p.append(self.predict(sample_fn())["probs"][0].numpy().astype(float))
        return dict(label=list(self.dataset.GRAPH_CLS.values()),
                    mean=np.mean(p, axis=0),
                    std=np.std(p, axis=0))

    @torch.no_grad()
    def quantatitive_baseline(self, **kwargs):
        return self.quantatitive(sample_fn=lambda: nx.gnp_random_graph(n=self.sampler.n, p=1/self.sampler.n, seed=self.seed),
                                 **kwargs)

    # TODO: do not rely on dataset for drawing
    @torch.no_grad()
    def evaluate(self, *args, show=False, connected=False, path=None, **kwargs):
        self.sampler.eval()
        G = self.sampler.sample(*args, **kwargs, seed=self.seed)
        if connected:
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
            self.print_graph_info(G)
            
        if show:
            self.show(G)
            # plt.savefig(path)
        return G

    def show(self, G, ax=None, path=None):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        pred = self.predict(G)
        logits = pred["logits"].mean(dim=0).tolist()
        probs = pred["probs"].mean(dim=0).tolist()
        print(f"{n=} {m=}")
        print(f"{logits=}")
        print(f"{probs=}")
        self.dataset.draw(G, ax=ax, path=path)

    def plot_networkx_graph(self, G):
        pass

    def save(self, G, cls_idx, root="result"):
        if isinstance(cls_idx, tuple):
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx[0]]}-{self.dataset.GRAPH_CLS[cls_idx[1]]}"
        else:
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx]}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        pickle.dump(G, open(f"{path}/{name}.pkl", "wb"))
        # self.show(G)
        # plt.savefig(f"{path}/{name}.png", bbox_inches="tight")
        # plt.show()

    def print_graph_info(self, G):
        print("Graph info: \n")
        print("Number of nodes: ", G.number_of_nodes())
        print("Number of edges: ", G.number_of_edges())
        print("Nodes: ", G.nodes.data())
        print("Edges: ", G.edges.data())
        print("\n")

    def load(self, id, root="result"):
        path = f"{root}/{self.dataset.name}/*"
        G = pickle.load(open(glob.glob(f"{path}/{id}.pkl")[0], "rb"))
        self.show(G)
        return G

    def evaluate_neg(self, *args, show_neg_edges=True, **kwargs):
        self.sampler.eval()
        neg_edge = self.sampler.sample(*args, **kwargs, seed=self.seed)
        G = nx.Graph(self.sampler.edges)
        G.remove_edges_from(neg_edge)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"{n=} {m=}")
        layout = nx.kamada_kawai_layout(G)
        if not show_neg_edges:
            nx.draw(G, pos=layout)
            plt.axis('equal')
            return G
        G.add_edges_from(neg_edge, edge_color='r', width=1)
        edge_color = [G[u][v].get('edge_color', 'k') for u, v in G.edges]
        width = [G[u][v].get('width', 1) for u, v in G.edges]
        nx.draw(G, pos=layout, edge_color=edge_color, width=width)
        plt.axis('equal')
        return G