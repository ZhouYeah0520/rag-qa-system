"""允许 `python -m rag_qa ...` 方式运行。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
