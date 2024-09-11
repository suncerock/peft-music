from typing import Optional, Literal

from transformers import AutoConfig

from .base import *
from .backbone.mert import MERTModel

from .modules.adapter import HubertEncoderLayerAdapter
from .modules.lora import LoRALinear

EVAL_LENGTH = 120000

class MERT(BaseModel):
    def __init__(
        self,
        exp: str,

        optim: Dict,

        encoder_dim: int = 768,
        encoder_depth: int = 12,

        freeze_feature: bool = False,
        freeze_encoder: bool = False,

        model_size: Literal["95M", "330M"] = "95M",
        ckpt_path: Optional[str] = "models/backbone/pytorch_model.bin",
        ):

        super().__init__(exp)
        self.inference_segment_length = EVAL_LENGTH

        self.freeze_feature = freeze_feature
        self.freeze_encoder = freeze_encoder

        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-{}".format(model_size), trust_remote_code=True)
        config.mask_time_prob = 0.
        config.mask_feature_prob = 0.
        self.mert = MERTModel(config)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.mert.load_state_dict(ckpt, strict=False)

        self.mert.encoder.layers = self.mert.encoder.layers[:encoder_depth]

        if self.freeze_feature:
            for param in self.mert.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.mert.feature_projection.parameters():
                param.requires_grad = False

        if self.freeze_encoder:
            for param in self.mert.encoder.parameters():
                param.requires_grad = False

        self.optim_cfg = optim
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(encoder_dim, N_OUTPUTS[exp])
        )

        self.loss_fn = ALL_LOSSES[exp]()

    def common_step(self, batch, test=False):
        x, y = batch["x"], batch["y"]

        emb = self.mert(x, output_hidden_states=False).last_hidden_state

        emb = aggregate_output_embedding(self.exp, emb)

        pred = self.proj(emb)

        loss_dict = dict()

        loss_pred = self.loss_fn(pred, y)
        loss_dict["loss/pred"] = loss_pred
        loss_dict["loss/total"] = loss_pred

        return loss_dict, pred

    def test_step(self, batch, batch_idx) -> Any:
        x, y = batch["x"], batch["y"]
        
        x = self.segment_inference_input(x)

        try:
            emb = self.mert(x, output_hidden_states=False).last_hidden_state
        except torch.cuda.OutOfMemoryError:
            emb = [self.mert(x[i].unsqueeze(dim=0), output_hidden_states=False).last_hidden_state for i in range(len(x))]
            emb = torch.concatenate(emb, dim=0)

        emb = aggregate_output_embedding(self.exp, emb)

        pred = self.proj(emb)
        pred = torch.sigmoid(pred)
        pred = pred.mean(dim=0, keepdim=True)

        loss_dict = dict()

        loss_pred = self.loss_fn(pred, y)
        loss_dict["loss/pred"] = loss_pred
        loss_dict["loss/total"] = loss_pred

        self.log_dict_prefix(loss_dict, "test")

        self.test_metrics.update(pred, batch["y"])

    def predict_step(self, x: torch.Tensor):
        x = self.segment_inference_input(x)

        with torch.no_grad():
            try:
                emb = self.mert(x, output_hidden_states=False).last_hidden_state
            except torch.cuda.OutOfMemoryError:
                emb = [self.mert(x[i].unsqueeze(dim=0), output_hidden_states=False).last_hidden_state for i in range(len(x))]
                emb = torch.concatenate(emb, dim=0)
            
            emb = aggregate_output_embedding(self.exp, emb)

            pred = self.proj(emb)
            pred = torch.sigmoid(pred)
            pred = pred.mean(dim=0, keepdim=True)

        # process the output
        output = convert_scores_to_outputs(self.exp, pred)
        return output

    def configure_optimizers(self) -> Any:
        self.optimizer_cfg = self.optim_cfg["optimizer"]
        self.scheduler_cfg = self.optim_cfg["scheduler"]
        return torch.optim.__dict__.get(self.optimizer_cfg["name"])(self.parameters(), **self.optimizer_cfg["args"])


class AdapterMERT(MERT):
    def __init__(
        self,

        exp: str,
        optim: Dict,
        
        encoder_dim: int = 768,
        encoder_depth: int = 12,

        bottleneck_dim: int = 16,
        
        model_size: Literal["95M", "330M"] = "95M",
        ckpt_path: Optional[str] = "models/backbone/pytorch_model.bin"
    ):
        super().__init__(
            exp,
            optim,

            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            
            freeze_feature=True,
            freeze_encoder=True,
            
            model_size=model_size,
            ckpt_path=ckpt_path
        )

        for i in range(len(self.mert.encoder.layers)):
            layer = self.mert.encoder.layers[i]
            
            for param in layer.layer_norm.parameters():
                param.requires_grad = True
            for param in layer.final_layer_norm.parameters():
                param.requires_grad = True

            self.mert.encoder.layers[i] = HubertEncoderLayerAdapter(layer, bottleneck_dim)


class LoRAMERT(MERT):
    def __init__(
        self,
        exp: str,

        optim: Dict,

        encoder_dim: int = 768,
        encoder_depth: int = 12,

        lora_rank_att: int = 2,
        lora_rank_ffn: int = 2,

        model_size: Literal["95M", "330M"] = "95M",
        ckpt_path: Optional[str] = "models/backbone/pytorch_model.bin"
    ):

        super().__init__(
            exp=exp,
            optim=optim,

            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            
            freeze_feature=True,
            freeze_encoder=True,
            
            model_size=model_size,
            ckpt_path=ckpt_path
        )
        for layer in self.mert.encoder.layers:
            self.intiailize_lora(
                layer, lora_rank_att=lora_rank_att, lora_rank_ffn=lora_rank_ffn)


    def intiailize_lora(self, model, lora_rank_att=None, lora_rank_ffn=None, lora_rank_conv=None):
        if lora_rank_att is not None:
            model.attention.q_proj = LoRALinear(model.attention.q_proj, lora_rank_att)
            model.attention.k_proj = LoRALinear(model.attention.k_proj, lora_rank_att)
            model.attention.v_proj = LoRALinear(model.attention.v_proj, lora_rank_att)
            model.attention.out_proj = LoRALinear(model.attention.out_proj, lora_rank_att)

        if lora_rank_ffn is not None:
            model.feed_forward.intermediate_dense = LoRALinear(model.feed_forward.intermediate_dense, lora_rank_ffn)
            model.feed_forward.output_dense = LoRALinear(model.feed_forward.output_dense, lora_rank_ffn)
