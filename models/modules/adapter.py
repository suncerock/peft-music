import math
import torch
import torch.nn as nn

from transformers.models.hubert.modeling_hubert import HubertEncoderLayer

from ..musicfm_modules.flash_conformer import Wav2Vec2ConformerEncoderLayer

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim) -> None:
        super().__init__()

        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, input_dim)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor):
        res = x

        x = self.down(x)
        x = self.act(x)
        x = self.up(x)

        return res + x


class Wav2Vec2ConformerEncoderLayerAdapter(nn.Module):
    def __init__(self, layer: Wav2Vec2ConformerEncoderLayer, bottleneck_dim=16) -> None:
        super().__init__()

        self.layer = layer

        self.ffn1_adapter = Adapter(layer.ffn1.output_dense.out_features, bottleneck_dim)
        self.ffn2_adapter = Adapter(layer.ffn2.output_dense.out_features, bottleneck_dim)
        self.attn_adapter = Adapter(layer.ffn1.output_dense.out_features, bottleneck_dim)

    def ffn1_forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer.ffn1_layer_norm(hidden_states)
        hidden_states = self.layer.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        return hidden_states

    def self_attention_forward(self, hidden_states, attention_mask, relative_position_embeddings, output_attentions):
        hidden_states, attn_weigts = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
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
        hidden_states = self.ffn1_adapter(hidden_states)
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.layer.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attention_forward(hidden_states, attention_mask, relative_position_embeddings, output_attentions)
        hidden_states = hidden_states + residual
        hidden_states = self.attn_adapter(hidden_states)

        # 3. Convolutional Layer
        hidden_states = self.convolution_forward(hidden_states)

        # 4. Feed-Forward 2 Layer
        hidden_states = self.ffn2_forward(hidden_states)
        hidden_states = self.ffn2_adapter(hidden_states)
        hidden_states = self.layer.final_layer_norm(hidden_states)

        return hidden_states, attn_weigts

class HubertEncoderLayerAdapter(nn.Module):
    def __init__(self, layer: HubertEncoderLayer, bottleneck_dim=16):
        super().__init__()

        self.layer = layer

        self.ffn_adapter = Adapter(layer.feed_forward.output_dense.out_features, bottleneck_dim)
        self.attn_adapter = Adapter(layer.feed_forward.output_dense.out_features, bottleneck_dim)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.layer.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.layer.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.attn_adapter(hidden_states)

        hidden_states = self.layer.layer_norm(hidden_states)
        hidden_states = hidden_states + self.layer.feed_forward(hidden_states)

        hidden_states = self.ffn_adapter(hidden_states)
        hidden_states = self.layer.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
