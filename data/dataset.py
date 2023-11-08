# -*- coding: utf-8 -*-
import os
import pathlib
from tqdm import tqdm
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from .collator import collator


class CadDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
    ):  
        assert split in ("train", "val", "test", "gan")
        path = pathlib.Path(root_dir)
        self.split = split
        self.files = []
        self.files_p = []
        self._get_filenames(path, filelist="{}.txt".format(split))


    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        with open(str(root_dir / f"{filelist}"), "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(root_dir.rglob(f"*[0-9].bin")):
            if x.stem in file_list:
                self.files.append(x)
        self.files = sorted(self.files, key=lambda name: int(os.path.basename(name)[0:-4]))
        print("Done loading {} files".format(len(self.files)))

        if (self.split == "train"):
            print(f"Loading data...")
            with open(str(root_dir / f"{filelist}"), "r") as f:
                file_list = ["{}_p".format(x.strip()) for x in f.readlines()]
            for x in tqdm(root_dir.rglob(f"*_p.bin")):
                if x.stem in file_list:
                    self.files_p.append(x)
            self.files_p = sorted(self.files_p, key=lambda name: int(os.path.basename(name)[0:-6]))
            print("Done loading {} files".format(len(self.files_p)))


    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        graph = graphfile[0][0]       
        dense_adj = graph.adj().to_dense().type(torch.int)
        N = graph.num_nodes()

        pyg_graph = PYGGraph()
        pyg_graph.graph = graph
        pyg_graph.node_data = graph.ndata["x"].type(FloatTensor)   #node_data[num_nodes, U_grid, V_grid, pnt_feature]
        pyg_graph.face_area = graph.ndata["y"].type(torch.int)     #face_area[num_nodes]
        pyg_graph.face_type = graph.ndata["z"].type(torch.int)     #face_type[num_nodes]
        pyg_graph.edge_data = graph.edata["x"].type(FloatTensor)
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)       
        pyg_graph.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        pyg_graph.edge_path = graphfile[1]["edges_path"]           # edge_input[num_nodes, num_nodes, max_dist, 1, U_grid, pnt_feature]
        pyg_graph.spatial_pos = graphfile[1]["spatial_pos"]        # spatial_pos[num_nodes, num_nodes]
        pyg_graph.d2_distance = graphfile[1]["d2_distance"]        # d2_distance[num_nodes, num_nodes, 64]
        pyg_graph.angle_distance = graphfile[1]["angle_distance"]  # angle_distance[num_nodes, num_nodes, 64]

        if ("commands_primitive" in graphfile[1]):
            pyg_graph.label_commands_primitive = graphfile[1]["commands_primitive"]
            pyg_graph.label_args_primitive = graphfile[1]["args_primitive"]
            pyg_graph.label_commands_feature = graphfile[1]["commands_feature"]
            pyg_graph.label_args_feature = graphfile[1]["args_feature"]
        else:
            pyg_graph.label_commands_primitive = torch.zeros([10])
            pyg_graph.label_args_primitive = torch.zeros([10, 11])
            pyg_graph.label_commands_feature = torch.zeros([12])
            pyg_graph.label_args_feature = torch.zeros([12, 12])
        if (self.split == "test"):
            pyg_graph.data_id = int(os.path.basename(file_path)[0:-4])
        return pyg_graph


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        fn = self.files[idx]
        sample = self.load_one_graph(fn)
        if (self.split == "train"):
            fn_p = self.files_p[idx]
            sample_p = self.load_one_graph(fn_p)
            return {"sample": sample,
                    "sample_p": sample_p
                    }
        else:
            return sample


    def _collate(self, batch):  #batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator(
            items=batch,
            split=self.split
        )


    def get_dataloader(self, batch_size, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=drop_last,
        )