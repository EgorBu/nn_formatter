import importlib
from typing import Dict, List, Tuple

import bblfsh
import numpy


from tokenizer.virtual_node import Position, VirtualNode


class CodeTokenizer:
    def __init__(self, language: str = "javascript", bblfsh_address: str = "0.0.0.0:9432"):
        """
        Construct a `CodeTokenizer`.

        :param language: Which language to extract features for.
        :param bblfsh_address: Address of bblfsh server.
        """
        self.language = language.lower()
        # import everything related to language
        self.tokens = importlib.import_module("tokenizer.langs.%s.tokens" % language)
        self.roles = importlib.import_module("tokenizer.langs.%s.roles" % language)
        try:
            self.token_unwrappers = importlib.import_module(
                "tokenizer..langs.%s.token_unwrappers" % language).TOKEN_UNWRAPPERS
        except ImportError:
            # It's normal for some languages not to have a token_unwrappers module.
            self.token_unwrappers = {}
        try:
            self.node_fixtures = importlib.import_module(
                "tokenizer.langs.%s.uast_fixers" % language).NODE_FIXTURES
        except ImportError:
            # It's normal for some languages not to have a uast_fixes module.
            self.node_fixtures = {}

        # Create instance of bblfsh client in case of bblfsh_address is not None.
        # If None - UAST has to be provided by client.
        if bblfsh_address is not None:
            self.client = bblfsh.BblfshClient(bblfsh_address)

    def tokenize_code(self, path: str, contents: str = None, root: bblfsh.Node = None) -> \
            Tuple[List[VirtualNode], Dict[int, bblfsh.Node]]:
        """
        Parse a file into a sequence of `VirtuaNode`-s and a mapping from VirtualNode to parent.

        Given the source text and the corresponding UAST this function compiles the list of
        `VirtualNode`-s and the parents mapping. That list of nodes equals to the original
        source text bit-to-bit after `"".join(n.value for n in nodes)`. `parents` map from
        `id(node)` to its parent `bblfsh.Node`.

        :param path: path to the file.
        :param contents: source file text, if not provided - path will be used to read content.
        :param root: UAST root node. If None - the file will be parsed using bblfsh client.
        :return: list of `VirtualNode`-s, the parents and root.
        """
        if contents is None:
            contents = self.client._get_contents(contents=contents, filename=path)
        if root is None:
            root = self.client.parse(filename=path, contents=contents).uast
        # build the line mapping
        contents = contents.decode()
        lines = contents.split("\n")
        line_offsets = numpy.zeros(len(lines) + 1, dtype=numpy.int32)
        pos = 0
        for i, line in enumerate(lines):
            line_offsets[i] = pos
            pos += len(line) + 1
        line_offsets[-1] = pos

        # walk the tree: collect nodes with assigned tokens and build the parents map
        node_tokens = []
        parents = {}
        queue = [root]
        while queue:
            node = queue.pop()
            if node.internal_type in self.node_fixtures:
                node = self.node_fixtures[node.internal_type](node)
            for child in node.children:
                parents[id(child)] = node
            queue.extend(node.children)
            if (node.token or node.start_position and node.end_position
                    and node.start_position != node.end_position and not node.children):
                node_tokens.append(node)
        node_tokens.sort(key=lambda n: n.start_position.offset)
        sentinel = bblfsh.Node()
        sentinel.start_position.offset = len(contents)
        sentinel.start_position.line = len(lines)
        node_tokens.append(sentinel)

        # scan `node_tokens` and fill the gaps with imaginary nodes
        result = []
        pos = 0
        parser = self.tokens.PARSER
        searchsorted = numpy.searchsorted
        for node in node_tokens:
            if node.start_position.offset < pos:
                continue
            if node.start_position.offset > pos:
                sumlen = 0
                diff = contents[pos:node.start_position.offset]
                for match in parser.finditer(diff):
                    positions = []
                    for suboff in (match.start(), match.end()):
                        offset = pos + suboff
                        line = searchsorted(line_offsets, offset, side="right")
                        col = offset - line_offsets[line - 1] + 1
                        positions.append(Position(offset, line, col))
                    token = match.group()
                    sumlen += len(token)
                    result.append(VirtualNode(token, *positions, path=path))
                assert sumlen == node.start_position.offset - pos, \
                    "missed some imaginary tokens: \"%s\"" % diff
            if node is sentinel:
                break
            result.extend(VirtualNode.from_node(node, contents, path, self.token_unwrappers))
            pos = node.end_position.offset
        return result, parents, root
