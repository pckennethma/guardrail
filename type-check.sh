mypy --disallow-untyped-defs -p nsyn
black . --exclude=notebooks --exclude=.venv
ruff --target-version=py310 --fix .