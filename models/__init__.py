from .musicfm import MusicFM, AdapterMusicFM, LoRAMusicFM
from .mert import MERT, AdapterMERT, LoRAMERT


ALL_MODELS = dict(
    musicfm=MusicFM,
    musicfm_adapter=AdapterMusicFM,
    musicfm_lora=LoRAMusicFM,
    

    mert=MERT,
    mert_adapter=AdapterMERT,
    mert_lora=LoRAMERT,
)