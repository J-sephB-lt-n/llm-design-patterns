from app.interfaces.memory_alg_protocol import MemoryAlg
from app.memory_algs.no_memory import NoMemory
from app.memory_algs.store_full_chat_history import StoreFullChatHistory

memory_algs: dict[str, type[MemoryAlg]] = {
    "No Memory": NoMemory,
    "Store Full Chat History": StoreFullChatHistory,
}
