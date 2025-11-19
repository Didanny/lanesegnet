#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .lane_attention import LaneAttention
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER.register_module()
class LaneSegNetTransformer(BaseModule):

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 points_num=1,
                 pts_dim=3,
                 **kwargs):
        super(LaneSegNetTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.points_num = points_num
        self.pts_dim = pts_dim
        self.fp16_enabled = False
        self.init_layers()

    def init_layers(self):
        self.reference_points = nn.Linear(self.embed_dims, self.pts_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, LaneAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)


    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                positional_query,
                content_query,
                geometric_query,
                bev_h,
                bev_w,
                reg_branches=None,
                cls_branches=None,
                geo_pointwise_branches=None,
                geo_global_branches=None,
                **kwargs):

        bs = mlvl_feats[0].size(0)
        
        # Use the passed queries directly (they will be projected in the decoder)
        query_pos = positional_query.unsqueeze(0).expand(bs, -1, -1)
        content_queries = content_query.unsqueeze(0).expand(bs, -1, -1)
        geometric_queries = geometric_query.unsqueeze(0).expand(bs, -1, -1, -1)
        reference_points = self.reference_points(query_pos) # nn.Linear [bs, num_query, embed_dims] -> [bs, num_query, pts_dim]

        # ident init: repeat reference points to num points
        reference_points = reference_points.repeat(1, 1, self.points_num)
        reference_points = reference_points.sigmoid()
        bs, num_qeury, _ = reference_points.shape
        reference_points = reference_points.view(bs, num_qeury, self.points_num, self.pts_dim)

        init_reference_out = reference_points

        # Switch num_query to the first dimension, batch to the second dimension
        content_queries = content_queries.permute(1, 0, 2) # [num_query, bs, embed_dims]
        query_pos = query_pos.permute(1, 0, 2) # [num_query, bs, embed_dims]
        geometric_queries = geometric_queries.permute(1, 0, 2, 3) # [num_query, bs, points_num, pts_dim]
        bev_embed = bev_embed.permute(1, 0, 2) # [bev_h*bev_w, bs, embed_dims]
        
        # self.decoder: LaneSegNetDecoder
        inter_content, inter_geo, inter_coords, inter_references = self.decoder(
            geometry_queries=geometric_queries,
            content_queries=content_queries,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            geo_pointwise_branches=geo_pointwise_branches,
            geo_global_branches=geo_global_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=content_queries.device),
            level_start_index=torch.tensor([0], device=content_queries.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_content, inter_geo, inter_coords, init_reference_out, inter_references_out
