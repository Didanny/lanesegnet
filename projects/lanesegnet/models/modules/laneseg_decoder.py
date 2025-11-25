#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LaneSegNetDecoder(TransformerLayerSequence):

    def __init__(self, 
                 *args, 
                 return_intermediate=False, 
                 pc_range=None, 
                 sample_idx=[0, 3, 6, 9], # num_ref_pts = len(sample_idx) * 2
                 pts_dim=3, 
                 **kwargs):
        super(LaneSegNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.sample_idx = sample_idx
        self.pts_dim = pts_dim

    def forward(self,
                content_queries,
                *args,
                query_pos=None,
                geometry_queries=None,
                reference_points=None,
                reg_branches=None,
                geo_pointwise_branches=None,
                geo_global_branches=None,
                key_padding_mask=None,
                **kwargs):
        
        # Initialize the current geometry guess from the input geometry_queries
        # geometry_queries: [num_query, bs, points_num, pts_dim] -> [num_query, bs, points_num*pts_dim]
        bs = content_queries.size(1)
        num_query = content_queries.size(0)
        embed_dims = content_queries.size(2) * 2
        
        # output = content_queries # [num_query, bs, embed_dims//2]
        intermediate_content = []
        intermediate_geometry = []
        intermediate_reference_points = []
        intermediate_coords = []
        lane_ref_points = reference_points[:, :, self.sample_idx * 2, :]
        
        # Init content_embeddings
        content_embeddings = content_queries # [num_query, bs, embed_dims//2]
        
        for lid, layer in enumerate(self.layers):
            # Project the geometry queries
            pointwise_embeddings = geo_pointwise_branches[lid](geometry_queries) # [num_query, bs, points_num, embed_dims]
            pointwise_embeddings = pointwise_embeddings.view(num_query, bs, -1) # [num_query, bs, points_num*embed_dims]
            geometry_embeddings = geo_global_branches[lid](pointwise_embeddings) # [num_query, bs, embed_dims//2]
            
            # Combine all queries: [content, geometry] -> [embed_dims]
            output = torch.cat([content_embeddings, geometry_embeddings], dim=-1)  # [num_query, bs, embed_dims]
            
            # BS NUM_QUERY NUM_LEVEL NUM_REFPTS 3
            reference_points_input = lane_ref_points[..., :2].unsqueeze(2)
            
            # Pass combined query through the transformer layer
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            
            # Batch-first dimension again
            output = output.permute(1, 0, 2) # [bs, num_query, embed_dims]

            if reg_branches is not None:
                # Split the query into content and geometry again
                content_embeddings, geometry_embeddings = torch.split(
                    output, embed_dims // 2, dim=-1) # [bs, num_query, embed_dims//2], [bs, num_query, embed_dims//2]
                
                # Duplicating input to keep branches exactly the same as baselines
                # AND to decouple geometry and content completely
                content_duplicated = torch.cat([content_embeddings, content_embeddings], dim=-1)
                geometry_duplicated = torch.cat([geometry_embeddings, geometry_embeddings], dim=-1)
                
                reg_center = reg_branches[0]
                reg_offset = reg_branches[1]

                # Project layer output to coordinate space to get residuals in logit space
                # This matches the baseline approach: unbounded residuals in logit space
                coord_residuals = reg_center[lid](geometry_duplicated) # [bs, num_query, points_num*pts_dim]
                bs, num_query, _ = coord_residuals.shape
                coord_residuals = coord_residuals.view(bs, num_query, -1, self.pts_dim) # [bs, num_query, points_num, pts_dim]
                
                # Work in logit space: add unbounded residuals directly to logit-space coordinates
                # This allows large corrections in early layers, like the baseline
                geometry_queries = geometry_queries.permute(1, 0, 2, 3) # [bs, num_query, points_num, pts_dim]
                new_coords_logit = coord_residuals + geometry_queries  # Direct addition in logit space
                
                # Convert to normalized space for use
                new_coords = new_coords_logit.sigmoid()  # [bs, num_query, points_num, pts_dim] in (0, 1)
                
                # Store normalized coords as reference_points for tracking
                reference_points = new_coords.detach()
                
                # Keep in logit space for next layer's recurrent connection
                geometry_queries = new_coords_logit

                # Denormalize polyline coordinates for offset addition
                # Use the normalized coordinates we just computed
                coord = new_coords.clone()  # [bs, num_query, points_num, pts_dim] in [0, 1]
                coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                if self.pts_dim == 3:
                    coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                centerline = coord.view(bs, num_query, -1).contiguous()

                # Calculate left and right lane lines from centerline + offset
                offset = reg_offset[lid](geometry_duplicated)
                left_laneline = centerline + offset
                right_laneline = centerline - offset
                
                # Sample new reference points from the lane lines
                left_laneline = left_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]
                right_laneline = right_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]

                lane_ref_points = torch.cat([left_laneline, right_laneline], axis=-2).contiguous().detach()

                # Normalize lane reference points back to [0, 1] range (After properly adding offset)
                lane_ref_points[..., 0] = (lane_ref_points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                lane_ref_points[..., 1] = (lane_ref_points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                if self.pts_dim == 3:
                    lane_ref_points[..., 2] = (lane_ref_points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

            if self.return_intermediate:
                intermediate_content.append(content_embeddings) # [bs, num_query, embed_dims//2]
                intermediate_geometry.append(geometry_embeddings) # [bs, num_query, embed_dims//2]
                intermediate_coords.append(new_coords) # [bs, num_query, points_num, pts_dim] - store normalized coords
                intermediate_reference_points.append(reference_points)
                
            # Set up the reused variables for the next layer
            # geometry_queries already in logit space from inverse_sigmoid above
            geometry_queries = geometry_queries.permute(1, 0, 2, 3) # [num_query, bs, points_num, pts_dim]
            content_embeddings = content_embeddings.permute(1, 0, 2) # [num_query, bs, embed_dims//2]

        if self.return_intermediate:
            return (
                torch.stack(intermediate_content),
                torch.stack(intermediate_geometry),
                torch.stack(intermediate_coords),
                torch.stack(intermediate_reference_points),
            )

        # Return final output if no intermediate outputs are needed
        return content_embeddings, geometry_embeddings, geometry_queries, reference_points


@TRANSFORMER_LAYER.register_module()
class CustomDetrTransformerDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(CustomDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
