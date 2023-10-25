# -*- coding: utf-8 -*-
import os
import pathlib
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from dgl.data.utils import load_graphs
from .collator import collator
from .utils import center_and_scale_pointcloud

class CadDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=False,
    ):  
        assert split in ("train", "val", "test", "gan")
        path = pathlib.Path(root_dir)
        self.split = split
        self.file_paths = []
        self.file_paths_main = []

        if split == "train":
            self._get_filenames(path, filelist="train.txt")
        elif split == "val":
            self._get_filenames(path, filelist="val.txt")
        elif split == "test":
            self._get_filenames(path, filelist="test.txt")
            self.file_paths = sorted(self.file_paths, key=lambda name: int(os.path.basename(name)[5:-4]))
        elif split == "gan":
            self._get_filenames(path, filelist="gan.txt")


    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        with open(str(root_dir / f"{filelist}"), "r") as f:
            file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(root_dir.rglob(f"*[0-9].bin")):
            if x.stem in file_list:
                self.file_paths.append(x)
        print("Done loading {} files".format(len(self.file_paths)))

        if (self.split == "train"):
            print(f"Loading data...")
            with open(str(root_dir / f"{filelist}"), "r") as f:
                file_list_main = [x.strip() + "_main" for x in f.readlines()]

            for x in tqdm(root_dir.rglob(f"*_main.bin")):
                if x.stem in file_list_main:  # 文件名不带后缀
                    self.file_paths_main.append(x)
            print("Done loading {} files".format(len(self.file_paths_main)))

            self.file_paths = sorted(self.file_paths, key=lambda name: int(os.path.basename(name)[0:-4]))
            self.file_paths_main = sorted(self.file_paths_main, key=lambda name: int(os.path.basename(name)[0:-9]))

    def load_one_graph(self, file_path):
        graphfile = load_graphs(str(file_path))
        
        graph = graphfile[0][0]       
        dense_adj = graph.adj().to_dense().type(torch.int)
        N = graph.num_nodes()  
        
        pyg_graph = PYGGraph()
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

        pyg_graph.graph = graph

        if(self.split == "train" or self.split == "val"):       
            pyg_graph.label_commands_main = graphfile[1]["commands_main"]
            pyg_graph.label_args_main = graphfile[1]["args_main"]
            pyg_graph.label_commands_sub = graphfile[1]["commands_sub"]
            pyg_graph.label_args_sub = graphfile[1]["args_sub"]

        if (self.split == "test"):
            pyg_graph.data_id = int(os.path.basename(file_path)[5:-4])

        return pyg_graph


    def center_and_scale(self, pnts):
        center, scale = center_and_scale_pointcloud(pnts)
        pnts[..., :3] -= center
        pnts[..., :3] *= scale
        

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        if (self.split == "train"):
            fn_ = self.file_paths_main[idx]
            sample_main = self.load_one_graph(fn_)
            return {"sample": sample, "sample_main": sample_main}
        else:
            return sample
        
    def _collate(self, batch):  #batch=({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
        return collator(
            batch,
            multi_hop_max_dist=8,  # multi_hop_max_dist: max distance of multi-hop edges 大于该值认为这两个节点没有关系，边编码为0
            spatial_pos_max=32,    # spatial_pos_max: max distance of multi-hop edges 大于该值认为这两个节点没有关系，空间编码降为0
            split=self.split
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )