import os 
import ipdb
import numpy as np
from math import sqrt
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers import LlamaConfig, LlamaModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from layers.embed import PositionalEmbedding
from models import GWNet


def compute_cosine_similarity(x, y, decimals=3):
    # Normalize the embeddings along the last dimension
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    
    # Compute the cosine similarity using the dot product of normalized vectors
    if len(y.shape) == 2:
        similarities = torch.einsum('bld,vd->blv', x_norm, y_norm)
        # similarities = torch.matmul(x_norm, y_norm.T)  
    elif len(y.shape) == 3:
        similarities = torch.einsum('bld,bvd->blv', x_norm, y_norm)
        # similarities = torch.matmul(x_norm, y_norm.transpose(1,2))
    elif len(y.shape) == 4:
        similarities = torch.einsum('bld,blvd->blv', x_norm, y_norm)
    else:
        raise ValueError('Unsupported dimension of y! Expected 2D or 3D tensor.')
    
    
    return similarities

class STEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, in_channel, embed_dim, seq_len, num_nodes, hod_time=24, norm_layer=None):
        super().__init__()
        self.c_in = in_channel
        self.patch_len = seq_len
        self.input_embedding = nn.Conv2d(in_channel,
                                        embed_dim,
                                        kernel_size=(1, self.patch_len),
                                        stride=(1, self.patch_len))

        self.node_embedding = nn.Parameter(torch.empty(num_nodes, embed_dim))

        self.hod_embedding = nn.Parameter(torch.empty(hod_time, embed_dim))
        self.dow_embedding = nn.Parameter(torch.empty(7, embed_dim))

        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        self.init_emb()

    def init_emb(self):
        nn.init.xavier_uniform_(self.hod_embedding)
        nn.init.xavier_uniform_(self.dow_embedding)
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Very long-term historical MTS with shape [B, T, N, c],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d]
        """
        B, T, N, C = x.shape

        # patch
        x_patch = self.input_embedding(x[...,:self.c_in].permute(0,3,2,1))
        x_patch = x_patch.permute(0,3,2,1).reshape(B, N, -1)

        # hour_of_day and day_of_week encodding
        x_hod = self.hod_embedding[
            (x[:, -1, :, self.c_in]).type(torch.LongTensor)
        ] 
        x_dow = self.dow_embedding[
            (x[:, -1, :, self.c_in+1]).type(torch.LongTensor)
        ]  # [B, N, D]

        x_time = x_hod + x_dow
        
        # node embedding 
        x_node = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        x_embed = x_patch + x_node + x_time

        # norm
        x_embed = self.norm_layer(x_embed)
        
        return x_embed

class Air_OURs(nn.Module):
    def __init__(self, config):
        super(Air_OURs, self).__init__()
        self.config = config
        # self.device = config.gpu
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model
        self.mask_attention = config.mask_attention
        self.prefix_alignment = config.prefix_alignment
        self.num_nodes = config.num_nodes
        self.T = config.tau

        if 'PEMS' in config.data:
            hod_time = 288
        elif 'AIR' in config.data:
            hod_time = 24
        else:
            hod_time = 48

        if config.backbone == 'gpt2':
            self.gpt2_config = GPT2Config.from_pretrained('pretrained_model/openai-community/gpt2/')
            self.gpt2_config.n_layer = config.llm_layers  
            self.gpt2_config.output_attentions = False 
            self.gpt2_config.output_hidden_states = False  

            self.backbone_llm = GPT2Model.from_pretrained(
                'pretrained_model/openai-community/gpt2/',
                trust_remote_code=True,
                local_files_only=True, 
                config=self.gpt2_config,
                # device_map={f'': self.device}
            )
            
            for i, (name, param) in enumerate(self.backbone_llm.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                elif 'mlp' in name and config.mlp == 1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.d_llm = self.gpt2_config.hidden_size # 768
            self.nlayer_llm = self.gpt2_config.n_layer # 6
            self.nheads_llm = self.gpt2_config.n_head # 6

        elif config.backbone == 'llama2':
            self.llama_config = LlamaConfig.from_pretrained('pretrained_model/meta-llama/Llama-2-7b-hf/')
            self.llama_config.num_hidden_layers = config.llm_layers 
            self.llama_config.output_attentions = False 
            self.llama_config.output_hidden_states = False 
        
            self.backbone_llm = LlamaModel.from_pretrained(
                'pretrained_model/meta-llama/Llama-2-7b-hf/',
                trust_remote_code=True,
                local_files_only=True, 
                config=self.llama_config,
                # device_map={f'': self.device}
            )

            for param in self.backbone_llm.parameters():
                param.requires_grad = False
            
            self.d_llm = self.llama_config.hidden_size # 4096
            self.nlayer_llm = self.llama_config.num_hidden_layers # 32 
            self.nheads_llm = self.llama_config.num_attention_heads # 32 

  
        self.st_embedding = STEmbedding(in_channel=config.feat_dims[0], embed_dim=config.d_model, seq_len=config.seq_len, num_nodes=config.num_nodes, hod_time=hod_time)
        
        self.input_projection_layer = nn.Conv2d(in_channels=self.d_model,
                                                out_channels=self.d_llm,
                                                kernel_size=(1, 1),
                                                bias=True)
        self.output_projection_layer = nn.Conv2d(in_channels=self.d_llm, 
                                                out_channels=self.seq_len, 
                                                kernel_size=(1,1),
                                                bias=True)
        self.rec_projection_layer = nn.Conv2d(in_channels=self.d_model,
                                            out_channels=self.seq_len,
                                            kernel_size=(1, 1),
                                            stride=(1, 1),
                                            bias=True)
        
        # pre-trained semantic word token embeddings
        if self.prefix_alignment:
            self.num_s_prefixes = config.num_pre_s
            self.num_t_prefixes = config.num_pre_t
            self.word_embeddings = self.backbone_llm.get_input_embeddings().weight
            self.mapping_s_layer = nn.Linear(self.word_embeddings.shape[0], config.word_size)
            self.mapping_t_layer = nn.Linear(self.word_embeddings.shape[0], config.word_size)
            self.length_predictor = nn.Sequential(
                nn.Linear(self.d_llm * config.num_nodes, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_s_prefixes)
                )

    def forward(self, history_data, adj=None):
        """Feed forward of our model.
        Args:
            history_data (torch.Tensor): historical data. shape: [B, T, N, 3] 24hour+7day
        Returns:
            torch.Tensor: prediction with shape [B, T, N, 1]
        """

        B, T, N, C = history_data.size()
        assert T == self.seq_len, 'input sequence length not equal to preset sequence length'
        
        # 1. spatial-temporal embedding
        inputs_embeds = self.st_embedding(history_data) # b, n, d_llm / B, d_llm,n,1
        inputs_embeds = inputs_embeds.permute(0, 2, 1).reshape(B, -1, N, 1)

        rec_x = self.rec_projection_layer(inputs_embeds)

        inputs_embeds = self.input_projection_layer(inputs_embeds).squeeze(-1).permute(0,2,1)
        
        # 2. align the work embedding
        alignment_loss = torch.tensor(0.)
        if self.prefix_alignment:
            combined_flat = inputs_embeds.reshape(B, -1)
            length_logits = self.length_predictor(combined_flat)
            optimal_length = torch.argmax(length_logits, dim=-1) + 1

            word_prototypes = self.mapping_s_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # [V, D]
            if adj == 1: return word_prototypes
            attention_mask = torch.ones(B, self.num_s_prefixes + N, device=inputs_embeds.device)
            
            prefix_prompts = []
            for i in range(B):
                len_i = optimal_length[i].item()
                prompt_prefix, alignment_loss = self.inputs_align_word_prototypes(inputs_embeds[i].unsqueeze(0), word_prototypes, len_i)
                prompt_prefix = F.pad(prompt_prefix, (0, 0, 0, self.num_s_prefixes - len_i))  # Pad to max_length
                prefix_prompts.append(prompt_prefix)
                attention_mask[i, len_i:self.num_s_prefixes] = 0
            # print(optimal_length)
            prefix_prompts = torch.cat(prefix_prompts, dim=0)
            inputs_embeds = torch.cat((prefix_prompts, inputs_embeds), dim=1)

 
        # 3. input the llm model
        if self.config.backbone == 'gpt2':
            dec_out = self.backbone_llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        else:
            dec_out = self.backbone_llm(inputs_embeds=inputs_embeds).last_hidden_state

        # 4. output results
        dec_out = dec_out[:, -N:].permute(0,2,1).unsqueeze(-1)
        outputs = self.output_projection_layer(dec_out)

        result = {
            'pred': outputs,
            'rec':rec_x,
            'align': alignment_loss
        }
        return result


    def inputs_align_word_prototypes(self, inputs_embeds, word_prototypes, num_prefixes):
        """
        Args:
            inputs_embeds: [B, L, D]
            word_prototypes: [V, D]
        Returns:
            mask: [nlayers x B x nheads x L x L] 
        """
        B, L, D = inputs_embeds.shape
        # Compute similarities
        align_similarities = compute_cosine_similarity(inputs_embeds, word_prototypes)# [B, L, V]
        
        # Select top relevant word prototypes based on similarity scores
        top_prototypes_indices = torch.topk(align_similarities, k=num_prefixes, dim=-1).indices

        # # Initialize the output tensor
        # output_indices = torch.zeros(B, num_prefixes, dtype=torch.long)
        # flattened_indices = top_prototypes_indices.view(B, -1)
        # for b in range(B):
        #     # find the frequency of each index 
        #     unique_indices, counts = torch.unique(flattened_indices[b], return_counts=True)
        #     sorted_indices = unique_indices[counts.argsort(descending=True)]
        #     sorted_indices_repeated = sorted_indices.repeat((num_prefixes // sorted_indices.size(0)) + 1)
            
        #     # Fill the output tensor with the most common indices
        #     output_indices[b] = sorted_indices_repeated[:num_prefixes]

        # Retrieve the top-K prompts
        prefix_prompts = word_prototypes[top_prototypes_indices.view(-1)].view(B, L, num_prefixes, -1)
        
        # Compute the topk word prototypes alignment_loss
        alignment_loss = compute_cosine_similarity(inputs_embeds, prefix_prompts)
        alignment_loss = - torch.mean(alignment_loss.mean(dim=-1).sum(dim=-1))
        prefix_prompts = prefix_prompts.mean(1)

        return prefix_prompts, alignment_loss
