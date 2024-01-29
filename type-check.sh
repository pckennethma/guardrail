mypy --disallow-untyped-defs -p nsyn && \
black . --exclude=notebooks --exclude=.venv,.lib && \
ruff --target-version=py310 --fix ./nsyn && \
ruff --target-version=py310 --fix ./scripts && \
# ignore lib/blip folder
cloc . --exclude-lang=CSV,JSON,Java,Maven,ReScript,MATLAB,Text,Markdown,YAML,INI,"Bourne Shell",TOML,Properties,XML,HTML,XML,XSLT,SVG --fullpath --match-d='/(nsyn|scripts|example_query|lib/fastmecenumeration)/'