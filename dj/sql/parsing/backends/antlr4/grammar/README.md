## Building

from `./grammar` you can run 
`docker pull any0ne22/antlr4:4.9.3`
to pull the required image then

`docker run -v `pwd`:/work any0ne22/antlr4:4.9.3 java -Xmx500M -cp /usr/local/lib/antlr4-tool.jar org.antlr.v4.Tool -Dlanguage=Python3 -o /work/generated SqlBase.g4`

this will generate the `./grammar/generated` directory`

alternatively, you can install locally which requires Java so probably don't want to install that vaporware

## Creating the diagram
Use https://bottlecaps.de/convert/ to go from ANTLR4 -> EBNF

Input the EBNF into https://bottlecaps.de/rr/ui