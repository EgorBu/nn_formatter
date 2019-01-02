"""
Module to change content of the file without changing the structure (UAST).

Algorithm:
1) code tokenization.
2) merge spaces, tabs, newlines into new IndentationNode.
3) find places and insert NoopNode.
4) find non UAST breaking mutations - delete, insert, replace
   Assumption - if one mutation doesn't change UAST, combination of several non UAST breaking
   mutations will not change UAST too.
5) create list with VirtualNodes and MutationNodes (nodes that contains several mutation options).
6) create generator that will combine mutations from several MutationNodes -> yield (list of
   generated VirtualNodes and list of VirtualNodes from initial dataset).

Example of usage:
```
import mutator
import tokenizer

# tokenize code
path = "/path/to/file.js"
js_tokenizer = tokenizer.CodeTokenizer("javascript")
tokens, parents, root = js_tokenizer.tokenize_code(path=path)

# find mutants
m = mutator.Mutator()
mutants = m.find_mutants(tokens, root)

# get initial code
print(mutator.get_initial(mutants))

# random mutation of code with the same UAST
print(mutator.get_sample(mutants))

# measure score
print(mutator.measure_distance(path=path, code_a=mutator.get_initial(mutants),
                               code_b=mutator.get_sample(mutants), return_seq=False))
```
"""
import difflib
from itertools import islice
import random
from typing import Iterator, List, Union

import bblfsh
import numpy

from tokenizer.tokenizer import CodeTokenizer
from tokenizer.virtual_node import Position, VirtualNode

INDENTATIOS = (" ", "\n", "\t")
QUOTES = ("'", '"')
LITERAL_STR = ("LITERAL", "STRING")
LITERAL_ID = bblfsh.role_id(LITERAL_STR[0])
STRING_ID = bblfsh.role_id(LITERAL_STR[1])


def is_indentation(node: VirtualNode):
    for ch in node.value:
        if ch not in INDENTATIOS:
            return False
    return True


def is_literal_string(token):
    if token.node is None:
        return False
    return LITERAL_ID in token.node.roles and STRING_ID in token.node.roles


class IndentationNode(VirtualNode):
    def __init__(self, node: VirtualNode):
        err = "Node should contain only indentation characters. Got '%s'"
        assert is_indentation(node), err % node.value
        self.value = node.value
        self.start = node.start
        self.end = node.end
        self.node = None
        self.path = node.path
        self.nodes = [node]

    def __add__(self, other: VirtualNode):
        err = "other should contain only indentation characters. Got '%s'"
        assert is_indentation(other), err % other.value
        assert self.path == other.path, "Path should be the same. Got '%s' and '%s'" % (self.path,
                                                                                        other.path)
        self.value += other.value
        self.end = other.end
        self.node.append(other)

    def __repr__(self) -> str:
        return ("IndentationNode(\"%s\", start=%s, end=%s, node=%s, path=\"%s\")" % (
            self.value.replace("\n", "\\n").replace("\t", "\\t"),
            tuple(self.start),
            tuple(self.end),
            id(self.node) if self.node is not None else "None",
            self.path))


class NoopNode(VirtualNode):
    def __repr__(self) -> str:
        return ("NoopNode(\"%s\", start=%s, end=%s, node=%s, path=\"%s\")" % (
                    self.value.replace("\n", "\\n").replace("\t", "\\t"),
                    tuple(self.start),
                    tuple(self.end),
                    id(self.node) if self.node is not None else "None",
                    self.path))


def check_uasts_are_equal(uast1: bblfsh.Node, uast2: bblfsh.Node) -> bool:
    """
    Check if 2 UASTs are identical or not in terms of nodes `roles`, `internal_type` and `token`.

    :param uast1: The bblfsh.Node of the first UAST to compare.
    :param uast2: The bblfsh.Node of the second UAST to compare.
    :return: A boolean equals to True if the 2 input UASTs are identical and False otherwise.
    """
    queue1 = [uast1]
    queue2 = [uast2]
    while queue1 or queue2:
        try:
            node1 = queue1.pop()
            node2 = queue2.pop()
        except IndexError:
            return False
        for child1, child2 in zip(node1.children, node2.children):
            if (child1.roles != child2.roles or child1.internal_type != child2.internal_type
                    or child1.token != child2.token):
                return False
        queue1.extend(node1.children)
        queue2.extend(node2.children)
    return True


def merge_indentation(tokens: List[VirtualNode]) -> \
        List[Union[VirtualNode, IndentationNode]]:
    """
    Merge spaces, tabs, newlines that are next to each other into IndentationNode.

    :param tokens: list of VirtualNodes after tokenization of source code.
    :return: list of VirtualNodes and IndentationNodes.
    """
    new_tokens = []
    ind_token = None
    for token in tokens:
        if ind_token is not None and is_indentation(token):
            ind_token += token
        elif ind_token is not None and not is_indentation(token):
            new_tokens.append(ind_token)
            ind_token = None
            new_tokens.append(token)
        elif ind_token is None and is_indentation(token):
            ind_token = IndentationNode(token)
        else:
            new_tokens.append(token)
    return new_tokens


def insert_noop(tokens: List[Union[VirtualNode, IndentationNode]]) -> \
        Union[VirtualNode, IndentationNode, NoopNode]:
    """
    Find places and insert NoopNode. Insertion should be done between
    :param tokens: list of tokens.
    :return: list of tokens extended by NoopNodes.
    """
    new_tokens = []
    if not len(tokens):
        return new_tokens
    # If first token is not IndentationNode - insert NoopNode.
    if type(tokens[0]) == VirtualNode:
        new_tokens.append(NoopNode(value="", start=Position(0, 1, 1),
                                   end=Position(0, 1, 1), path=tokens[0].path))
    for token, next_token in zip(tokens, islice(tokens, 1, None)):
        new_tokens.append(token)
        # (quote & literal) or (literal & quote) - don't insert NoopNode
        if (token.value in QUOTES and token.node is None) and \
                is_literal_string(next_token):
            continue
        elif (next_token.value in QUOTES and next_token.node is None) and \
                is_literal_string(token):
            continue
        # (Any & IndentationNode) or (IndentationNode & Any) - don't insert NoopNode
        elif is_indentation(token) or is_indentation(next_token):
            continue
        assert token.end == next_token.start
        new_tokens.append(NoopNode(value="", start=token.end, end=token.end, path=token.path))
    new_tokens.append(next_token)
    if tokens[-1].node is not None:
        new_tokens.append(NoopNode(value="", start=tokens[-1].end, end=tokens[-1].end,
                                   path=tokens[-1].path))
    return new_tokens


class MutantNode:
    """Class to store initial state of node and available mutations."""
    def __init__(self, initial_val: Union[IndentationNode, NoopNode]):
        self.initial_val_ = initial_val
        self.mutants_ = set([initial_val])  # initially only this option is available

    @property
    def initial_val(self):
        return self.initial_val_

    @property
    def mutants(self, ):
        return self.mutants_

    def add_mutant(self, mut):
        self.mutants_.add(mut)

    @property
    def value(self):
        # return random sample from mutants
        return random.sample(self.mutants_, 1)[0]


def populate_mutants_(mutant_node: Union[IndentationNode, NoopNode, str], n_trials: int = 4,
                      max_rep: int = 3, max_ins: int = 3, enforce_empty: bool = True) -> \
        Iterator[str]:
    """
    Function to randomly populate mutants given node to mutate.
    1) yield "" if it's not NoopNode
    2) introduce 'small' mutations
    while n < n_trials:
        randomly select position
        randomly select mutation (delete, insert something else several times n_ins < max_ins,
                                  repeat: randomly choose n_rep < max_rep)
        update & yield result

    :param mutant_node: node that should be mutated.
    :param n_trials: number of trials for mutation.
    :param max_rep: max number of repeats.
    :param max_ins: max number of insertions.
    :param enforce_empty: yield empty string if val is not empty.
    :yield mutants one by one.
    """
    val = mutant_node if isinstance(mutant_node, str) else mutant_node.value
    if val and enforce_empty:
        yield ""  # 1) yield "" if it's not NoopNode
        n_trials -= 1
    actions = ["insert"]
    if val:
        # these actions available only for
        actions.append("delete")
        actions.append("repeat")

    while n_trials:
        n_trials -= 1
        # select action to do
        action = numpy.random.choice(actions)
        # select position
        if val:
            pos = numpy.random.randint(0, len(val))
        else:
            pos = 0
        before, after = val[:pos], val[pos:]
        if action == "insert":
            insertion_val = numpy.random.choice(INDENTATIOS)
            n_times = numpy.random.randint(1, max_ins)
            yield before + insertion_val * n_times + after
        elif action == "delete":
            if len(after) == 0:
                yield before[:-1]
            else:
                yield before + after[1:]
        else:
            # repeat
            n_repeats = numpy.random.randint(0, max_rep)
            if len(after) == 0:
                yield before[:-1] + before[-1] * n_repeats
            else:
                yield before + after[0] * n_repeats + after[1:]


def populate_mutants(mutant_node: Union[IndentationNode, NoopNode, str], n_trials: int = 4,
                     max_rep: int = 3, max_ins: int = 3, depth: int = 2) -> Iterator[str]:
    """
    Function to populate mutants given node to mutate.
    Ex: `depth = 2` means: take first level mutants, yield them and generate new mutants from them.

    :param mutant_node: node that should be mutated.
    :param n_trials: number of trials for mutation.
    :param max_rep: max number of repeats.
    :param max_ins: max number of insertions.
    :param depth: depth of mutant generation.
    :yield mutants one by one.
    """
    # TODO: find elegant way to make depth a parameter
    mutants = [mutant_node]
    for _ in range(depth):
        new_mutants = []
        for node in mutants:
            for mutant in populate_mutants_(mutant_node=node, n_trials=n_trials, max_rep=max_rep,
                                            max_ins=max_ins):
                yield mutant
                new_mutants.append(mutant)
        mutants = new_mutants


class Mutator:
    """Mutator of content with preserving the same (U)AST structure."""
    def __init__(self, n_trials: int = 4, max_rep: int = 2, max_ins: int = 2,
                 max_mutants: int = 10, bblfsh_address: str = "0.0.0.0:9432"):
        """
        Initialize mutator.

        :param n_trials: number of trials for mutation.
        :param max_rep: max number of repeats.
        :param max_ins: max number of insertions.
        :param max_mutants: max number of mutants to collect.
        :param depth: max number of mutants to collect.
        """
        self.n_trials = n_trials
        self.max_rep = max_rep
        self.max_ins = max_ins
        self.max_mutants = max_mutants
        self.bblfsh_address = bblfsh_address
        self.client = bblfsh.BblfshClient(endpoint=bblfsh_address)

    def find_mutants(self, tokens, root):
        """
        This function should check several available mutations for each NoopNode & IndentationNode.
        As result it will give list
        :param tokens: list of tokens after `CodeTokenizer.tokenize_code`.
        :param root: root of UAST.
        :return: list of tokens where `IndentationNode`s & `NoopNode`s are replaced with
                 MutationNode.
        """
        tokens_ = insert_noop(merge_indentation(tokens))
        path = tokens[0].path
        new_tokens = []
        for token in tokens_:
            if type(token) in (IndentationNode, NoopNode):
                new_tokens.append(MutantNode(initial_val=token.value))
            else:
                new_tokens.append(token)
        assert len(new_tokens) == len(tokens_)
        # TODO: parallelization here
        for ind in range(len(new_tokens)):
            if isinstance(new_tokens[ind], MutantNode):
                for mutant in populate_mutants(new_tokens[ind].initial_val, max_ins=self.max_ins,
                                               n_trials=self.n_trials, max_rep=self.max_rep):
                    if mutant in new_tokens[ind].mutants:
                        # duplicated mutant - skip
                        continue
                    # TODO: check uast
                    new_content = []
                    for i, t in enumerate(new_tokens):
                        if i == ind:
                            new_content.append(mutant)
                        else:
                            if isinstance(t, MutantNode):
                                new_content.append(t.initial_val)
                            else:
                                new_content.append(t.value)
                    new_content = "".join(new_content)
                    new_root = self.client.parse(filename=path,
                                                 contents=new_content.encode("utf-8")).uast
                    if check_uasts_are_equal(root, new_root):
                        new_tokens[ind].add_mutant(mutant)
                    if len(new_tokens[ind].mutants) >= self.max_mutants:
                        break
        return new_tokens


def get_sample(mutant_tokens):
    return "".join(t.value for t in mutant_tokens)


def get_initial(mutant_tokens):
    return "".join(t.value if not isinstance(t, MutantNode) else t.initial_val
                   for t in mutant_tokens)


def measure_distance(path: str, code_a: Union[str, bytes], code_b: Union[str, bytes],
                     return_seq: bool = True) -> Union[List[float], float]:
    """
    Measure distance between 2 pieces of code.
    Algorithm
    1) tokenize -> merge_indentation -> insert_noop
    2) iterate over 2 list of tokens and measure common ratio between them (if only one of them is
       IndentationNode - ratio is 0, if both - measure score using  `SequenceMatcher.ratio`)
    TLDR: The more - the better.

    :param path: path to file.
    :param code_a: piece of code A.
    :param code_b: piece of code B.
    :param return_seq: if return sequence of scores or average.
    :return: sequence of ratios or average ratio.
    """
    try:
        code_a = code_a.encode("utf-8")
    except AttributeError:
        pass
    try:
        code_b = code_b.encode("utf-8")
    except AttributeError:
        pass
    ct = CodeTokenizer()
    sm = difflib.SequenceMatcher()
    tokens_a = insert_noop(merge_indentation(ct.tokenize_code(path=path, contents=code_a)[0]))
    tokens_b = insert_noop(merge_indentation(ct.tokenize_code(path=path, contents=code_b)[0]))
    assert len(tokens_a) == len(tokens_b)
    scores = []
    for ta, tb in zip(tokens_a, tokens_b):
        is_ind_a = isinstance(ta, IndentationNode)
        is_ind_b = isinstance(tb, IndentationNode)
        if is_ind_a and is_ind_b:
            sm.set_seq1(ta.value)
            sm.set_seq2(tb.value)
            scores.append(sm.ratio())
        elif is_ind_a or is_ind_b:
            scores.append(0)
    if return_seq:
        return scores
    return sum(scores) / len(scores)
