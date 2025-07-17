
# ChatBot Memory Explorer


## Dev Notes

- Memory algorithms are defined in `app/memory_algs/` and must follow the protocol defined in `app/interfaces/memory_alg_protocol.py`
- Algorithm parameters are automatically parsed from the `__init__()` method of the `MemoryAlg` class and this is used to dynamically generate streamlit input widgets for algorithm setup.

