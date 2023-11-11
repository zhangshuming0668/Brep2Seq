# -*- coding: utf-8 -*-
import json
import sys

sys.path.append('..')
from .macro import *


def vec_to_json(main_commands, main_param, sub_commands, sub_param, output_file):
    json_root = {}
    main_feature = []
    sub_feature = []

    # output primitives
    for i in range(np.size(main_commands, 0)):
        a_feature = {}
        if (main_commands[i] == BOX_IDX):
            a_feature["feature_type"] = "box"
            feature_param = {}
            feature_param['L1'] = int(main_param[i][0])
            feature_param['L2'] = int(main_param[i][1])
            feature_param['L3'] = int(main_param[i][2])
            feature_param['T_x'] = int(main_param[i][4])
            feature_param['T_y'] = int(main_param[i][5])
            feature_param['T_z'] = int(main_param[i][6])
            feature_param['Q_0'] = int(main_param[i][7])
            feature_param['Q_1'] = int(main_param[i][8])
            feature_param['Q_2'] = int(main_param[i][9])
            feature_param['Q_3'] = int(main_param[i][10])
            a_feature["param"] = feature_param
            main_feature.append(a_feature)

        elif (main_commands[i] == PRISM_IDX):
            a_feature["feature_type"] = "prism"
            feature_param = {}
            feature_param['L1'] = int(main_param[i][0])
            feature_param['L2'] = int(main_param[i][1])
            feature_param['E'] = int(main_param[i][3])
            feature_param['T_x'] = int(main_param[i][4])
            feature_param['T_y'] = int(main_param[i][5])
            feature_param['T_z'] = int(main_param[i][6])
            feature_param['Q_0'] = int(main_param[i][7])
            feature_param['Q_1'] = int(main_param[i][8])
            feature_param['Q_2'] = int(main_param[i][9])
            feature_param['Q_3'] = int(main_param[i][10])
            a_feature["param"] = feature_param
            main_feature.append(a_feature)

        elif (main_commands[i] == CYLINDER_IDX):
            a_feature["feature_type"] = "cylinder"
            feature_param = {}
            feature_param['L1'] = int(main_param[i][0])
            feature_param['L2'] = int(main_param[i][1])
            feature_param['T_x'] = int(main_param[i][4])
            feature_param['T_y'] = int(main_param[i][5])
            feature_param['T_z'] = int(main_param[i][6])
            feature_param['Q_0'] = int(main_param[i][7])
            feature_param['Q_1'] = int(main_param[i][8])
            feature_param['Q_2'] = int(main_param[i][9])
            feature_param['Q_3'] = int(main_param[i][10])
            a_feature["param"] = feature_param
            main_feature.append(a_feature)

        elif (main_commands[i] == CONE_IDX):
            a_feature["feature_type"] = "cone"
            feature_param = {}
            feature_param['L1'] = int(main_param[i][0])
            feature_param['L2'] = int(main_param[i][1])
            feature_param['T_x'] = int(main_param[i][4])
            feature_param['T_y'] = int(main_param[i][5])
            feature_param['T_z'] = int(main_param[i][6])
            feature_param['Q_0'] = int(main_param[i][7])
            feature_param['Q_1'] = int(main_param[i][8])
            feature_param['Q_2'] = int(main_param[i][9])
            feature_param['Q_3'] = int(main_param[i][10])
            a_feature["param"] = feature_param
            main_feature.append(a_feature)

        elif (main_commands[i] == SPHERE_IDX):
            a_feature["feature_type"] = "sphere"
            feature_param = {}
            feature_param['L1'] = int(main_param[i][0])
            feature_param['T_x'] = int(main_param[i][4])
            feature_param['T_y'] = int(main_param[i][5])
            feature_param['T_z'] = int(main_param[i][6])
            a_feature["param"] = feature_param
            main_feature.append(a_feature)

    # output features
    for i in range(np.size(sub_commands, 0)):
        a_feature = {}
        if (sub_commands[i] == RECT_SLOT_IDX):
            a_feature["feature_type"] = "rect_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == TRI_SLOT_IDX):
            a_feature["feature_type"] = "tri_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == CIR_SLOT_IDX):
            a_feature["feature_type"] = "cir_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == RECT_PSG_IDX):
            a_feature["feature_type"] = "rect_psg"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == TRI_PSG_IDX):
            a_feature["feature_type"] = "tri_psg"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == HEXA_PSG_IDX):
            a_feature["feature_type"] = "hexa_psg"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == HOLE_IDX):
            a_feature["feature_type"] = "hole"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == RECT_STEP_IDX):
            a_feature["feature_type"] = "rect_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == TSIDE_STEP_IDX):
            a_feature["feature_type"] = "tside_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid_s1'] = int(sub_param[i][9])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == SLANT_STEP_IDX):
            a_feature["feature_type"] = "slant_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == RECT_B_STEP_IDX):
            a_feature["feature_type"] = "rect_b_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == TRI_STEP_IDX):
            a_feature["feature_type"] = "tri_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == CIR_STEP_IDX):
            a_feature["feature_type"] = "cir_step"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['dep'] = int(sub_param[i][10])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == RECT_B_SLOT_IDX):
            a_feature["feature_type"] = "rect_b_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == CIR_B_SLOT_IDX):
            a_feature["feature_type"] = "cir_b_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == U_B_SLOT_IDX):
            a_feature["feature_type"] = "u_b_slot"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == RECT_PKT_IDX):
            a_feature["feature_type"] = "rect_pkt"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == KEY_PKT_IDX):
            a_feature["feature_type"] = "key_pkt"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['wid'] = int(sub_param[i][6])
            feature_param['len'] = int(sub_param[i][7])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == TRI_PKT_IDX):
            a_feature["feature_type"] = "tri_pkt"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == HEXA_PKT_IDX):
            a_feature["feature_type"] = "hexa_pkt"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == O_RING_IDX):
            a_feature["feature_type"] = "o_ring"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == B_HOLE_IDX):
            a_feature["feature_type"] = "b_hole"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == CHAMFER_IDX):
            a_feature["feature_type"] = "chamfer"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

        elif (sub_commands[i] == FILLET_IDX):
            a_feature["feature_type"] = "fillet"
            feature_param = {}
            feature_param['x1'] = int(sub_param[i][0])
            feature_param['y1'] = int(sub_param[i][1])
            feature_param['z1'] = int(sub_param[i][2])
            feature_param['x2'] = int(sub_param[i][3])
            feature_param['y2'] = int(sub_param[i][4])
            feature_param['z2'] = int(sub_param[i][5])
            feature_param['rad'] = int(sub_param[i][8])
            a_feature["param"] = feature_param
            sub_feature.append(a_feature)

    json_root["principal_primitive"] = main_feature
    json_root["detail_feature"] = sub_feature

    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(json_root, fp, indent=4)