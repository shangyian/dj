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

## OSS Spark Grammar

The lexer and grammar file from the master OSS Spark branch is stored in `oss_spark_grammar`.
They've been trimmed to remove all of the global settings checks and `@member` java methods.
You can generate them using the following commands, making sure to generate the lexer first.

```
antlr4 -Dlanguage=Python3 -visitor SqlBaseLexer.g4 -o generated
antlr4 -Dlanguage=Python3 -visitor SqlBaseParser.g4 -o generated
```

Then in the `../utils.py` file, uncomment the following two lines.

```
# Uncomment the following two lines and comment the two lines above to use the lexer and parser generated from the OSS Spark g4 file
# from dj.sql.parsing.backends.antlr4.grammar.oss_spark_grammar.generated.SqlBaseLexer import SqlBaseLexer
# from dj.sql.parsing.backends.antlr4.grammar.oss_spark_grammar.generated.SqlBaseParser import SqlBaseParser
```

## Creating the diagram
Use https://bottlecaps.de/convert/ to go from ANTLR4 -> EBNF

Input the EBNF into https://bottlecaps.de/rr/ui
