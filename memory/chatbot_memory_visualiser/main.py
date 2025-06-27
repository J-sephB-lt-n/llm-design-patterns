import json

from app.memory_algs import memory_alg_registry

print(
    json.dumps(
        memory_alg_registry,
        indent=4,
        default=str,
    ),
)
