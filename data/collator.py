# -*- coding: utf-8 -*-
import torch
import numpy as np
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


def collator(items, multi_hop_max_dist=10, spatial_pos_max=20, split="train"):  #items({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
    
    if split=="train":
        #total_CAD-----------------------------------------------------------------
        items_total = [
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
                item["sample"].label_commands_main,
                item["sample"].label_args_main,
                item["sample"].label_commands_sub,
                item["sample"].label_args_sub
            )
            for item in items
        ]
        #main_CAD------------------------------------------------------------------
        items_main = [
            (
                item["sample_main"].graph,
                item["sample_main"].node_data,  #node_data[num_nodes, U_grid, V_grid, pnt_feature]
                item["sample_main"].face_area,  
                item["sample_main"].face_type,              
                item["sample_main"].edge_data,  #edge_data[num_edges, U_grid, pnt_feature]
                item["sample_main"].in_degree,
                item["sample_main"].attn_bias,
                item["sample_main"].spatial_pos,
                item["sample_main"].d2_distance,
                item["sample_main"].angle_distance,
                item["sample_main"].edge_path[:, :, :multi_hop_max_dist],  #[num_nodes, num_nodes, max_dist]   
                item["sample_main"].label_commands_main,
                item["sample_main"].label_args_main,
                item["sample_main"].label_commands_sub,
                item["sample_main"].label_args_sub
            )
            for item in items
        ]
        items = items_total + items_main
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
            label_commandses_main,
            label_argses_main,
            label_commandses_sub,
            label_argses_sub
        ) = zip(*items)  #解压缩
        
    elif split=="val":
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
                item.label_commands_main,
                item.label_args_main,
                item.label_commands_sub,
                item.label_args_sub
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
            label_commandses_main,
            label_argses_main,
            label_commandses_sub,
            label_argses_sub
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
            data_ids
        ) = zip(*items)  #解压缩

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        
    max_node_num = max(i.size(0) for i in node_datas)  #计算这批数据中图节点的最大数量 
    max_edge_num = max(i.size(0) for i in edge_datas)
    max_dist = max(i.size(-1) for i in edge_paths)  #计算节点间的最大距离 针对某些图的max_dist都小于multi_hop_max_dist的情况
    max_dist = max(max_dist, multi_hop_max_dist)

    #对数据进行打包并返回, 将各数据调整到同一长度，以max_node_num为准  
    #图长度掩码
    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])
    
    #边长度掩码
    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])
    
    #节点特征
    node_data = torch.cat([i for i in node_datas])  #node_datas(batch_size, [num_nodes, U_grid, V_grid, pnt_feature])

    # face_area = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_areas])
    face_area = torch.cat([i for i in face_areas])

    # face_type = torch.cat([pad_face_unsqueeze(i, max_node_num) for i in face_types])
    face_type = torch.cat([i for i in face_types])
    
    #边特征
    edge_data = torch.cat([i for i in edge_datas])  #edge_datas(batch_size, [num_edges, U_grid, pnt_feature])元组
    
    #边编码输入
    edge_path = torch.cat(     #edges_paths(batch_size, [num_nodes, num_nodes, max_dist])
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
    ).long()
    
    #注意力矩阵
    attn_bias = torch.cat(      #attn_bias(batch_size, [num_nodes+1, num_nodes+1]) 多了一个全图的虚拟节点
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
   
    #空间编码
    spatial_pos = torch.cat(   #spatial_pos(batch_size, [num_nodes, num_nodes])
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )    
    d2_distance = torch.cat(
        [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distancees]
    )
    angle_distance = torch.cat(
        [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distancees]
    )
    
    #中心性编码
    # in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]) #in_degree(batch_size, [num_nodes])
    in_degree = torch.cat([i for i in in_degrees])

    batched_graph = dgl.batch([i for i in graphs])

    if (split=="train" or split=="val"):
        #标签数据
        batched_label_commands_main = torch.cat([torch.unsqueeze(i, dim=0) for i in label_commandses_main])
        batched_label_args_main = torch.cat([torch.unsqueeze(i, dim=0) for i in label_argses_main])
        batched_label_commands_sub = torch.cat([torch.unsqueeze(i, dim=0) for i in label_commandses_sub])
        batched_label_args_sub = torch.cat([torch.unsqueeze(i, dim=0) for i in label_argses_sub])

        batch_data = dict(
            padding_mask = padding_mask,       #[batch_size, max_node_num]
            edge_padding_mask = edge_padding_mask,  #[batch_size, max_edge_num]

            graph=batched_graph,
            node_data = node_data,             #[total_node_num, U_grid, V_grid, pnt_feature]  
            face_area = face_area,             #[batch_size, max_node_num] / [total_node_num]
            face_type = face_type,             #[batch_size, max_node_num] / [total_node_num]
            edge_data = edge_data,             #[total_edge_num, U_grid, pnt_feature]
            
            in_degree = in_degree,             #[batch_size, max_node_num]
            out_degree = in_degree,            #[batch_size, max_node_num] #无向图
            attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
            spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
            d2_distance = d2_distance,         #[batch_size, max_node_num, max_node_num, 64]
            angle_distance = angle_distance,   #[batch_size, max_node_num, max_node_num, 64]
            edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充
            
            label_commands_main = batched_label_commands_main,
            label_args_main = batched_label_args_main,
            label_commands_sub = batched_label_commands_sub,
            label_args_sub = batched_label_args_sub
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
            out_degree = in_degree,            #[batch_size, max_node_num] #无向图
            attn_bias = attn_bias,             #[batch_size, max_node_num+1, max_node_num+1]
            spatial_pos = spatial_pos,         #[batch_size, max_node_num, max_node_num]
            d2_distance = d2_distance,         #[batch_size, max_node_num, max_node_num, 32]
            angle_distance = angle_distance,   #[batch_size, max_node_num, max_node_num, 32]
            edge_path = edge_path,             #[batch_size, max_node_num, max_node_num, max_dist] 空位用-1填充

            id=data_ids
        )
   
    return batch_data
