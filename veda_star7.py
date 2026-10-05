from .vendor.veda.nodes import VedaSparseAttention, register_model_folder

register_model_folder()
NODE_CLASS_MAPPINGS = {"Star7VedaSparseAttention": VedaSparseAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"Star7VedaSparseAttention": "MiniMax H3 VEDA 稀疏注意力 - Star7"}
