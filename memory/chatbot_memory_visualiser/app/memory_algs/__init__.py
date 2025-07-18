from app.interfaces.memory_alg_protocol import MemoryAlg
from app.memory_algs.no_memory import NoMemory
from app.memory_algs.recursive_summarisation import RecursiveSummarisation
from app.memory_algs.store_full_chat_history import StoreFullChatHistory
from app.memory_algs.vector_memory import VectorMemory

memory_algs: dict[str, type[MemoryAlg]] = {
    "No Memory": NoMemory,
    "Recursive Summarisation": RecursiveSummarisation,
    "Store Full Chat History": StoreFullChatHistory,
    "Vector Memory": VectorMemory,
}
