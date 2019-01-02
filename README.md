# NN code formatter
Experiment with NN-based source code formatter.

Automatically learn source code formatting style - WIP for: `\n`, space, `\t`, newline.

## Pipeline
1) [Mutation step](#Mutation-algorithm)  - prepare mutants that will allow easily introduce random changes in source code without breaking UAST.
So next step algorithm will have `X` (modified code) and `Y` (initial code).
2) Learning step - train ML algorithm to reconstruct initial source code from mutated source code.
3) [Quality measurement step](#Scoring-algorithm) - given 2 pieces of source code (`Y_ground_truth` & `Y_predicted`) (with the same UAST structure) measure quality of indentations reconstruction.

## Mutation algorithm
TLDR: algorithm to find code with the same UAST structure and identifiers/literals but with different indentations (`\n`, space, `\t`, newline).
#### 1) Tokenization:
TLDR: `code`-> {`tokens`, `parents`, `UAST`} -> `"".join(tokens)` -> `code`.

 It means that you could split your code into stream of tokens (some of them will have specific UAST node) with UAST & information about parent for each node in UAST.
#### 2) Create `IndentationNode` (first possible type of mutation point):
Merge spaces, tabs, newlines that are next to each other together into `IndentationNode`.
#### 3) Insert `NoopNode` (second possible type of mutation point):
Places to insert - between every 2 tokens except:
1) one of them is `IndentationNode`.
2) one of them is quote (`'`/`"`) and other string literal.
#### 4) Find non UAST breaking mutations:
```
for each mutation point:
    check several hypotesis:
        select random position P in mutation point
        apply randomly one of next transformation:
        * replace full mutation point with empty string ""
        * insert and repeat one of indentation symbols ('\n', space, '\t') into position P
        * delete character at position P
        * repeat random character at position P
        keep mutation if it doesn't break UAST
```
Assumption - if one mutation doesn't change UAST, combination of several non UAST breaking mutations will not change UAST too.

After this step is possible to generate code with the same UAST but different indentations.
#### 5) Result: {`tokens` + `mutant tokens`, `parents`, `UAST`}
And functionality to:
1) generate random piece of code with the same UAST.
2) reconstruct initial code.

## Scoring algorithm
Input:
* Code A/B: raw code or list of tokens with `IndentationNode` & `NoopNode`.

Algorithm:
```
tokenize if needed
scores = []
for token_a, token_b in zip(tokens_a, tokens_b):
    if none of them are `IndentationNode` & `NoopNode` -> do nothing
    elif one token is `NoopNode` and another is `IndentationNode` -> scores.append(0)
    elif both are `IndentationNode` -> scores.append(similarity_ratio(token_a, token_b))
```

