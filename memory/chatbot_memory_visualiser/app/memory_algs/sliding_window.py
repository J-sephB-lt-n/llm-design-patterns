"""
Implementation of a memory algorithm which just stores the past X tokens/messages
"""

from app.interfaces.memory_alg_protocol import MemoryAlg
from app.memory_algs import register_memory_alg


@register_memory_alg()
class SlidingWindowMemory(MemoryAlg): ...
