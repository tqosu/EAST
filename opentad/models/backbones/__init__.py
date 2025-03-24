from .backbone_wrapper import BackboneWrapper
from .r2plus1d_tsp import ResNet2Plus1d_TSP
from .re2tal_swin import SwinTransformer3D_inv
from .re2tal_slowfast import ResNet3dSlowFast_inv
from .vit import VisionTransformerCP
from .vit_adapter import VisionTransformerAdapter
# dropout random
from .vit_adapter3 import VisionTransformerAdapter3
# pooling move to beginning and last
from .vit_adapter4 import VisionTransformerAdapter4
# tube dropout
from .vit_adapter3_1 import VisionTransformerAdapter3_1

# from .InternVideo2.single_modality.models.internvideo2 import InternVideo2
# from .InternVideo2.single_modality.models.internvideo2Adapter import InternVideo2Adapter
from .vit_ladder import VisionTransformerLadder
from .vit_ladder3 import VisionTransformerLadder3

# use side connect, but not use adapter
from .vit_adapterM import VisionTransformerAdapterM

# use side connect and use adapter
from .vit_adapterM1 import VisionTransformerAdapterM1

# remove 
from .vit_adapter1 import VisionTransformerAdapter1
# remove 
from .vit_ladder1 import VisionTransformerLadder1
__all__ = [
    "BackboneWrapper",
    "ResNet2Plus1d_TSP",
    "SwinTransformer3D_inv",
    "ResNet3dSlowFast_inv",
    "VisionTransformerCP",
    "VisionTransformerAdapter",
    "InternVideo2",
    "InternVideo2Adapter",
    "VisionTransformerLadder",
    "VisionTransformerAdapter1",
    "VisionTransformerLadder1",
    "VisionTransformerAdapter3",
    "VisionTransformerLadder3",
    "VisionTransformerAdapter4",
    "VisionTransformerAdapter3_1",
]
