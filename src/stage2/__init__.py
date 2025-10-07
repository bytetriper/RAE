from .models.lightningDiT import LightningDiT
from .models.DDT import DiTwDDTHead
from typing import Callable, Dict, Optional, Type, Union
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class Stage2ModelProtocol(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_cfg(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_autoguidance(self, *args: Any, **kwargs: Any) -> Any: ...



# STAGE2_ARCHS: Dict[str, Callable] = {}
# __all__ = ["STAGE2_ARCHS", "register_stage2"]


# def _add_to_registry(name: str, func: Callable) -> Callable:
#     if name in STAGE2_ARCHS and STAGE2_ARCHS[name] is not func:
#         raise ValueError(f"Stage2 func: '{name}' is already registered.")
#     STAGE2_ARCHS[name] = func
#     return func


# def register_stage2(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Union[Callable[[Callable], Callable], Callable]:
#     """Register a function in ``STAGE2_ARCHS``.

#     Can be used either as ``@register_stage2()`` (optionally passing ``name``) or
#     via ``register_stage2(my_function)`` after the function definition.
#     """

#     def decorator(inner_func: Callable) -> Callable:
#         func_name = name or inner_func.__name__
#         return _add_to_registry(func_name, inner_func)

#     if func is None:
#         return decorator

#     return decorator(func)

# @register_stage2()
# def DiTXL(token_dim: int, input_size: int) -> LightningDiT:
#     model = LightningDiT(
#         input_size=input_size,
#         patch_size=1,
#         in_channels=token_dim,
#         hidden_size=1152,
#         depth=28,
#         num_heads=16,
#         mlp_ratio=4.0,
#         class_dropout_prob=0.1,
#         num_classes=1000,
#     )
#     return model

# @register_stage2()
# def DDTXL(token_dim: int, input_size: int) -> DiTwDDTHead:
#     model = DiTwDDTHead(
#         input_size=input_size,
#         patch_size=1,
#         in_channels=token_dim,
#         hidden_size=[1152, 2048],
#         depth=[28, 2],
#         num_heads=[16, 16],
#         mlp_ratio=4.0,
#         class_dropout_prob=0.1,
#         num_classes=1000,
#     )
#     return model

# @register_stage2()
# def DDTS(token_dim: int, input_size: int) -> DiTwDDTHead:
#     model = DiTwDDTHead(
#         input_size=input_size,
#         patch_size=1,
#         in_channels=token_dim,
#         hidden_size=[384, 2048],
#         depth=[12, 2],
#         num_heads=[6, 16],
#         mlp_ratio=4.0,
#         class_dropout_prob=0.1,
#         num_classes=1000,
#     )
#     return model




