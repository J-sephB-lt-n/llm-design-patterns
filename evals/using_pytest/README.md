
# LLM evals using pytest

Lightweight model evaluations can live alongside the standard software tests in pytest

e.g.

```bash
tests/
├── end2end/
│   ├── end2end_test1_name.py
│   └── ...
├── integration/
│   ├── integration_test1_name.py
│   └── ...
├── llm_evals/                       < ---------- this one
│   ├── llm_eval1_name.py
│   └── ...
└── unit/
    ├── unit_test1_name.py
    └── ...
```

Some notes:

- pytest "fixtures" are resources which are created once and shared by multiple tests (e.g. environment variables, datasets, network connections etc.)
- Where you have many evals, rather put them into multiple independent test_*.py scripts
  - They can share setup config by giving them a shared `conftest.py` script (e.g. all test scripts can use the same LLM config supplied once)
  - You can have multiple conftest.py files (each applies to the test folder it's in)
- Not shown here, but pytest can be set up to write the test results to an arbitrary location (e.g. to an experiment tracking platform such as MLFlow or wandb)

To run the tests:
```bash
cd evals/using_pytest/
uv run pytest test_evals/ --model gpt-4o --temp 0 # run all tests in this folder
uv run pytest test_evals/test_example_llm_eval_1.py # run all tests in this script
```
