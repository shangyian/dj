## Building

Install antlr generator tool and antlr python runtime
```
pip install antlr4-tools
pip install antlr4-python3-runtime
```

from `./grammar` you can run 
```
antlr4 -Dlanguage=Python3 -visitor SqlBase.g4 -o generated
```

this will generate the `./grammar/generated` directory`

## Creating the diagram
Use https://bottlecaps.de/convert/ to go from ANTLR4 -> EBNF

Input the EBNF into https://bottlecaps.de/rr/ui
