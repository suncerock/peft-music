import torch
import torch.nn as nn

from transformers.models.hubert.modeling_hubert import HubertEncoderLayer

from ..musicfm_modules.flash_conformer import Wav2Vec2ConformerEncoderLayer


class Wav2Vec2ConformerEncoderLayerPrefix(nn.Module):
    def __init__(self, layer: Wav2Vec2ConformerEncoderLayer, num_prefix=32, bottleneck_dim=512) -> None:
        super().__init__()

        hidden_dim = layer.ffn1.output_dense.out_features
        self.num_prefix = num_prefix

        self.wte = nn.Embedding(num_prefix, hidden_dim)
        self.control_trans = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, hidden_dim * 2)
        )

        self.layer = layer

    def ffn1_forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer.ffn1_layer_norm(hidden_states)
        hidden_states = self.layer.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        return hidden_states

    def self_attention_forward(self, hidden_states, attention_mask, relative_position_embeddings, output_attentions):
        hidden_states = self.layer.self_attn_layer_norm(hidden_states)
        
        input_tokens = torch.arange(self.num_prefix).long().unsqueeze(0).expand(1, -1).to(hidden_states.device)
        embs = self.wte(input_tokens)
        kv_prefix = self.control_trans(embs)

        kv_prefix = torch.tile(kv_prefix, [hidden_states.shape[0]] + [1 for _ in range(len(hidden_states.shape) - 1)])
        
        hidden_states, attn_weigts = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
            kv_prefix=kv_prefix
        )
        hidden_states = self.layer.self_attn_dropout(hidden_states)
        return hidden_states, attn_weigts

    def convolution_forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer.conv_module(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def ffn2_forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer.ffn2_layer_norm(hidden_states)
        hidden_states = self.layer.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        relative_position_embeddings = None,
        output_attentions: bool = False,
    ):
        # 1. Feed-Forward 1 layer
        hidden_states = self.ffn1_forward(hidden_states)
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states, attn_weigts = self.self_attention_forward(hidden_states, attention_mask, relative_position_embeddings, output_attentions)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        hidden_states = self.convolution_forward(hidden_states)

        # 4. Feed-Forward 2 Layer
        hidden_states = self.ffn2_forward(hidden_states)
        hidden_states = self.layer.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts

class HubertEncoderLayerPrefix(nn.Module):
    def __init__(self, layer: HubertEncoderLayer, num_prefix=32, bottleneck_dim=512):
        super().__init__()

        hidden_dim = layer.feed_forward.output_dense.out_features
        self.num_prefix = num_prefix

        self.wte = nn.Embedding(num_prefix, hidden_dim)
        self.control_trans = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, hidden_dim * 2)
        )

        self.layer = layer

        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states

        input_tokens = torch.arange(self.num_prefix).long().unsqueeze(0).expand(1, -1).to(hidden_states.device)
        embs = self.wte(input_tokens)
        kv_prefix = self.control_trans(embs)

        kv_prefix = torch.tile(kv_prefix, [hidden_states.shape[0]] + [1 for _ in range(len(hidden_states.shape) - 1)])

        hidden_states, attn_weights, _ = self.layer.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, kv_prefix=kv_prefix
        )
        hidden_states = self.layer.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer.layer_norm(hidden_states)
        hidden_states = hidden_states + self.layer.feed_forward(hidden_states)

        hidden_states = self.layer.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs