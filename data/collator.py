# -*- coding: utf-8 -*-
import torch
import dgl
import sys
sys.path.append('..')
from models.modules.module_utils.macro import *

def pad_mask_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):  #x[num_nodes]
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_face_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_d2_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 32], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_ang_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 32], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)
     
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):  #x[num_nodes, num_nodes, max_dist]
    xlen1, xlen2, xlen3 = x.size() 
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, multi_hop_max_dist=8, spatial_pos_max=32, split="train"):  #items({PYGGraph_1, PYGGraph_1_p}, {PYGGraph_2, PYGGraph_2_p}, ..., batchsize)
    
    if split == "train":
        # primitives & feature
        items_0 = [
            (
                item["sample"].graph,
                item["sample"].node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item["sample"].face_area,  
                item["sample"].face_type, 
                item["sample"].edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item["sample"].in_degree,
                item["sample"].attn_bias,
                item["sample"].spatial_pos,
                item["sample"].d2_distance,
                item["sample"].angle_distance,
                item["sample"].edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]           
                item["sample"].label_commands_primitive,
                item["sample"].label_args_primitive,
                item["sample"].label_commands_feature,
                item["sample"].label_args_feature
            )
            for item in items
        ]
        # primitive
        items_1 = [
            (
                item["sample_p"].graph,
                item["sample_p"].node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item["sample_p"].face_area,
                item["sample_p"].face_type,
                item["sample_p"].edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item["sample_p"].in_degree,
                item["sample_p"].attn_bias,
                item["sample_p"].spatial_pos,
                item["sample_p"].d2_distance,
                item["sample_p"].angle_distance,
                item["sample_p"].edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]
                item["sample_p"].label_commands_primitive,
                item["sample_p"].label_args_primitive,
                item["sample_p"].label_commands_feature,
                item["sample_p"].label_args_feature
            )
            for item in items
        ]
        items = items_0 + items_1
        (
            graphs,
            node_datas,
            face_areas,
            face_types,
            edge_datas,
            in_degrees,
            attn_biases,
            spatial_poses, 
            d2_distancees,
            angle_distancees,
            edge_paths,
            label_commands_primitives,
            label_args_primitives,
            label_commands_features,
            label_args_features
        ) = zip(*items)
        
    elif split == "val" or split == "gan":
        items = [
            (
                item.graph,
                item.node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item.face_area,
                item.face_type,
                item.edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item.in_degree,
                item.attn_bias,
                item.spatial_pos,
                item.d2_distance,
                item.angle_distance,
                item.edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]   
                item.label_commands_primitive,
                item.label_args_primitive,
                item.label_commands_feature,
                item.label_args_feature
            )
            for item in items
        ]
        (
            graphs,
            node_datas,
            face_areas,
            face_types,
            edge_datas,
            in_degrees,
            attn_biases,
            spatial_poses, 
            d2_distancees,
            angle_distancees,
            edge_paths,
            label_commands_primitives,
            label_args_primitives,
            label_commands_features,
            label_args_features
        ) = zip(*items)  #解压缩

    else:
        items = [
            (
                item.graph,
                item.node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item.face_area,
                item.face_type,
                item.edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item.in_degree,
                item.attn_bias,
                item.spatial_pos,
                item.d2_distance,
                item.angle_distance,
                item.edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]
                item.label_commands_primitive,
                item.label_args_primitive,
                item.label_commands_feature,
                item.label_args_feature,
                item.data_id
            )
            for item in items
        ]
        (
            graphs,
            node_datas,
            face_areas,
            face_types,
            edge_datas,
            in_degrees,
            attn_biases,
            spatial_poses, 
            d2_distancees,
            angle_distancees,
            edge_paths,
            label_commands_primitives,
            label_args_primitives,
            label_commands_features,
            label_args_features,
            data_ids
        ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        
    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)
    max_dist = max(max_dist, multi_hop_max_dist)

    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])

    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])

    node_data = torch.cat([i for i in node_datas])  #node_datas(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])

    face_area = torch.cat([i for i in face_areas])

    face_type = torch.cat([i for i in face_types])

    edge_data = torch.cat([i for i in edge_datas])  #edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组

    edge_path = torch.cat(     #edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )

    spatial_pos = torch.cat(   #spatial_pos(batch_size, [num_nodes, num_nodes])
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )    
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distancees]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distancees]
    )

    in_degree = torch.cat([i for i in in_degrees])

    batched_graph = dgl.batch([i for i in graphs])

    # sequence labels
    batched_label_commands_primitive = torch.cat([torch.unsqueeze(i, dim=0) for i in label_commands_primitives])
    batched_label_args_primitive = torch.cat([torch.unsqueeze(i, dim=0) for i in label_args_primitives])
    batched_label_commands_feature = torch.cat([torch.unsqueeze(i, dim=0) for i in label_commands_features])
    batched_label_args_feature = torch.cat([torch.unsqueeze(i, dim=0) for i in label_args_features])

    if (split == "train" or split == "val" or split == "gan"):
        batch_data = dict(
            padding_mask = padding_mask,       #[batch_size, max_node_num]
            edge_padding_mask = edge_padding_mask,  #[batch_size, max_edge_num]

            graph=batched_graph,
            node_data = node_data,             #[total_node_num, U_grid, V_grid, pnt_feature]  
            face_area = face_area,             #[batch_size, max_node_num] / [total_node_num]
            face_type = face_type,             #[batch_size, max_node_num] / [total_node_num]
            edge_data = edge_data,             #[total_edge_num, U_grid, pnt_feature]
            
            in_degree = in_degree,             #[batch_size, max_node_num]
            out_degree = in_degree,            #[batch_size, max_node_num]
            attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
            spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
            d2_distance = d2_distance,         #[batch_size, max_node_num, max_node_num, 64]
            angle_distance = angle_distance,   #[batch_size, max_node_num, max_node_num, 64]
            edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充
            
            label_commands_primitive = batched_label_commands_primitive,
            label_args_primitive = batched_label_args_primitive,
            label_commands_feature = batched_label_commands_feature,
            label_args_feature = batched_label_args_feature
        )

    else:
        data_ids = torch.tensor([i for i in data_ids])
        batch_data = dict(
            padding_mask = padding_mask,       #[batch_size, max_node_num]
            edge_padding_mask = edge_padding_mask,  #[batch_size, max_edge_num]

            graph=batched_graph,
            node_data = node_data,             #[total_node_num, U_grid, V_grid, pnt_feature]  
            face_area = face_area,             #[batch_size, max_node_num] / [total_node_num]
            face_type = face_type,             #[batch_size, max_node_num] / [total_node_num]
            edge_data = edge_data,             #[total_edge_num, U_grid, pnt_feature]
            
            in_degree = in_degree,             #[batch_size, max_node_num]
            out_degree = in_degree,            #[batch_size, max_node_num]
            attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
            spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
            d2_distance = d2_distance,         #[batch_size, max_node_num, max_node_num, 32]
            angle_distance = angle_distance,   #[batch_size, max_node_num, max_node_num, 32]
            edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist]

            label_commands_primitive=batched_label_commands_primitive,
            label_args_primitive=batched_label_args_primitive,
            label_commands_feature=batched_label_commands_feature,
            label_args_feature=batched_label_args_feature,

            id=data_ids
        )
   
    return batch_data
