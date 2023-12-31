mypy --disallow-untyped-defs -p nsyn && \
black . --exclude=notebooks --exclude=.venv && \
ruff --target-version=py310 --fix . && \
# ignore lib/blip folder
cloc . --exclude-lang=CSV,JSON,Java,Maven,ReScript,MATLAB,Text,Markdown,YAML,INI,"Bourne Shell",TOML,Properties,XML,HTML,XML,XSLT,SVG --fullpath --not-match-d="lib/blip","./.*","models","datasets"