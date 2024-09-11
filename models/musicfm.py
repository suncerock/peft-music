from .base import *
from .backbone.musicfm import MusicFM25Hz

from .modules.adapter import Wav2Vec2ConformerEncoderLayerAdapter
from .modules.lora import LoRAConv1d, LoRALinear

EVAL_LENGTH_DEFAULT = 30 * 24000
EVAL_LENGTH = dict(
    mtg_top50=719725,
    mtg_genre=719725,
    mtg_instrument=719725,
    mtg_moodtheme=719725,
    
    mtat=698976
)

class MusicFM(BaseModel):
    def __init__(
        self,
        exp: str,

        optim: Dict,

        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,

        freeze_conv: bool = True,
        freeze_conformer: bool = False,

        stat_path: str = "./data/fma_classic_stats.json",
        model_path: str = "./data/musicfm_25hz_FMA_330m_500k.pt",
        ):

        super().__init__(exp)

        self.optim_cfg = optim
        self.inference_segment_length = EVAL_LENGTH.get(self.exp, EVAL_LENGTH_DEFAULT)

        self.musicfm = MusicFM25Hz(
            conv_dim=conv_dim, encoder_dim=encoder_dim, encoder_depth=encoder_depth, mask_hop=0.4, mask_prob=0.6, is_flash=True,
            stat_path=stat_path, model_path=model_path
        )

        self.encoder_depth = encoder_depth
        self.freeze_conv = freeze_conv
        self.freeze_conformer = freeze_conformer

        if self.freeze_conv:
            for param in self.musicfm.conv.parameters():
                param.requires_grad = False

        if self.freeze_conformer:
            for param in self.musicfm.conformer.parameters():
                param.requires_grad = False

        self.musicfm.linear = nn.Identity()  # Not used
        del self.musicfm.cls_token  # This is not used in the MusicFM

        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(encoder_dim, N_OUTPUTS[exp])
        )

        self.loss_fn = ALL_LOSSES[exp]()

    def common_step(self, batch, test=False):
        x, y = batch["x"], batch["y"]

        emb = self.musicfm.get_latent(x, layer_ix=self.encoder_depth)
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
            emb = self.musicfm.get_latent(x, layer_ix=self.encoder_depth)
        except torch.cuda.OutOfMemoryError:
            emb = [self.musicfm.get_latent(x[i].unsqueeze(dim=0), layer_ix=self.encoder_depth) for i in range(len(x))]
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
                emb = self.musicfm.get_latent(x, layer_ix=self.encoder_depth)
            except torch.cuda.OutOfMemoryError:
                emb = [self.musicfm.get_latent(x[i].unsqueeze(dim=0), layer_ix=self.encoder_depth) for i in range(len(x))]
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


class AdapterMusicFM(MusicFM):
    def __init__(
        self,
        exp: str,

        optim: Dict,

        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,

        bottleneck_dim=64,

        stat_path: str = "./data/fma_classic_stats.json",
        model_path: str = "./data/musicfm_25hz_FMA_330m_500k.pt",
    ):
        super().__init__(
            exp=exp,
            optim=optim,

            conv_dim=conv_dim,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            
            freeze_conv=True,
            freeze_conformer=True,
            
            stat_path=stat_path,
            model_path=model_path
        )

        for i in range(len(self.musicfm.conformer.layers)):
            layer = self.musicfm.conformer.layers[i]
            
            for param in layer.self_attn_layer_norm.parameters():
                param.requires_grad = True
            for param in layer.ffn1_layer_norm.parameters():
                param.requires_grad = True
            for param in layer.ffn2_layer_norm.parameters():
                param.requires_grad = True
            for param in layer.final_layer_norm.parameters():
                param.requires_grad = True

            self.musicfm.conformer.layers[i] = Wav2Vec2ConformerEncoderLayerAdapter(layer, bottleneck_dim)


class LoRAMusicFM(MusicFM):
    def __init__(
        self,
        exp: str,

        optim: Dict,

        conv_dim: int = 512,
        encoder_dim: int = 1024,
        encoder_depth: int = 12,

        lora_rank_att: int = 2,
        lora_rank_ffn: int = 2,
        lora_rank_conv: int = 2,

        stat_path: str = "./data/fma_classic_stats.json",
        model_path: str = "./data/musicfm_25hz_FMA_330m_500k.pt",
    ):

        super().__init__(
            exp=exp,
            optim=optim,

            conv_dim=conv_dim,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            
            freeze_conv=True,
            freeze_conformer=True,
            
            stat_path=stat_path,
            model_path=model_path
        )
        for layer in self.musicfm.conformer.layers:
            self.intiailize_lora(
                layer, lora_rank_att=lora_rank_att, lora_rank_ffn=lora_rank_ffn, lora_rank_conv=lora_rank_conv)


    def intiailize_lora(self, model, lora_rank_att=None, lora_rank_ffn=None, lora_rank_conv=None):
        if lora_rank_att is not None:
            model.self_attn.linear_q = LoRALinear(model.self_attn.linear_q, lora_rank_att)
            model.self_attn.linear_k = LoRALinear(model.self_attn.linear_k, lora_rank_att)
            model.self_attn.linear_v = LoRALinear(model.self_attn.linear_v, lora_rank_att)
            model.self_attn.linear_out = LoRALinear(model.self_attn.linear_out, lora_rank_att)

        if lora_rank_ffn is not None:
            model.ffn1.intermediate_dense = LoRALinear(model.ffn1.intermediate_dense, lora_rank_ffn)
            model.ffn1.output_dense = LoRALinear(model.ffn1.output_dense, lora_rank_ffn)
            model.ffn2.intermediate_dense = LoRALinear(model.ffn2.intermediate_dense, lora_rank_ffn)
            model.ffn2.output_dense = LoRALinear(model.ffn2.output_dense, lora_rank_ffn)

        if lora_rank_conv is not None:
            model.conv_module.pointwise_conv1 = LoRAConv1d(model.conv_module.pointwise_conv1, lora_rank_conv)
            model.conv_module.pointwise_conv2 = LoRAConv1d(model.conv_module.pointwise_conv2, lora_rank_conv)
