"""Defines VirtualNode - a class which backs any source code token."""
from typing import Callable, Iterable, Mapping, NamedTuple, Tuple, Union

import bblfsh


class Position(NamedTuple("Position", (("offset", int), ("line", int), ("col", int)))):
    """
    Data class to hold position information of virtual nodes.

    `line` and `col` are 1-based to match UAST, `offset` is 0-based.
    """

    @staticmethod
    def from_bblfsh_position(position: bblfsh.Position) -> "Position":
        """Create a lookout.style.format.virtual_node.Position from a bblfsh.Position."""
        line = position.line
        col = position.col
        if hasattr(position, "offset"):
            offset = position.offset
        elif line == 1 and col == 1:
            offset = 0
        else:
            raise ValueError("Offset missing.")
        return Position(offset, line, col)


class VirtualNode:
    """
    Represent either a real UAST node or an imaginary token.

    The class instance should be considered as read-only.
    """

    def __init__(self, value: str, start: Position, end: Position,
                 *, node: bblfsh.Node = None,  is_accumulated_indentation: bool=False,
                 path: str = None) -> None:
        """
        Construct a VirtualNode.

        :param value: Text of the token.
        :param start: Starting position of the token.
        :param end: Ending position of the token.
        :param node: Corresponding UAST node (if exists).
        :param is_accumulated_indentation: Marks virtual node as node with accumulated indentation.
        :param path: Path to related file. Useful for debugging.
        """
        assert start.line >= 1 and start.col >= 1, "start line and column are 1-based like UASTs"
        assert end.line >= 1 and end.col >= 1, "end line and column are 1-based like UASTs"
        # assert value or start.offset < end.offset, "bad positions"
        self.value = value
        self.start = start
        self.end = end
        self.node = node
        self.is_accumulated_indentation = is_accumulated_indentation
        self.path = path

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return ("VirtualNode(\"%s\", start=%s, end=%s, node=%s, path=\"%s\")" % (
                    self.value.replace("\n", "\\n").replace("\t", "\\t"),
                    tuple(self.start),
                    tuple(self.end),
                    id(self.node) if self.node is not None else "None",
                    self.path))

    def __eq__(self, other: "VirtualNode") -> bool:
        return (self.value == other.value
                and self.start == other.start
                and self.end == other.end
                and self.node == other.node
                and self.path == other.path)

    def copy(self) -> "VirtualNode":
        """Produce a full clone of the node."""
        return VirtualNode(
            self.value, self.start, self.end, node=self.node,
            is_accumulated_indentation=self.is_accumulated_indentation, path=self.path)

    @staticmethod
    def from_node(node: bblfsh.Node, file: str, path: str,
                  token_unwrappers: Mapping[str, Callable[[str], Tuple[str, str]]],
                  ) -> Iterable["VirtualNode"]:
        """
        Initialize the VirtualNode from a UAST node. Takes into account prefixes and suffixes.

        :param node: UAST node.
        :param file: File contents.
        :param path: File path.
        :param token_unwrappers: Mapping from bblfsh internal types to functions to unwrap tokens.
        :return: New VirtualNode-s.
        """
        outer_token = file[node.start_position.offset:node.end_position.offset]
        if not node.token:
            yield VirtualNode(outer_token,
                              Position.from_bblfsh_position(node.start_position),
                              Position.from_bblfsh_position(node.end_position),
                              node=node, path=path)
            return
        node_token = node.token
        if node.internal_type in token_unwrappers:
            node_token, outer_token = token_unwrappers[node.internal_type](outer_token)
        start_offset = outer_token.find(node_token)
        assert start_offset >= 0, (
            "Couldn't find the token in the specified position:\nNode role: %s\nParsed form: “%s”"
            "\nRaw form: “%s”\nStart position: %d, %d, %d\nEnd position: %d, %d, %d" % (
                node.internal_type, node_token, outer_token, node.start_position.offset,
                node.start_position.line, node.start_position.col, node.end_position.offset,
                node.end_position.line, node.end_position.col))
        start_pos = node.start_position.offset + start_offset
        if start_offset:
            yield VirtualNode(outer_token[:start_offset],
                              Position(offset=node.start_position.offset,
                                       line=node.start_position.line,
                                       col=node.start_position.col),
                              Position(offset=start_pos,
                                       line=node.start_position.line,
                                       col=node.start_position.col + start_offset),
                              path=path)
        end_offset = start_offset + len(node_token)
        end_pos = start_pos + len(node_token)
        yield VirtualNode(node_token,
                          Position(offset=start_pos,
                                   line=node.start_position.line,
                                   col=node.start_position.col + start_offset),
                          Position(offset=end_pos,
                                   line=node.end_position.line,
                                   col=node.start_position.col + end_offset),
                          node=node, path=path)
        if end_pos < node.end_position.offset:
            yield VirtualNode(outer_token[end_offset:],
                              Position(offset=end_pos,
                                       line=node.end_position.line,
                                       col=node.start_position.col + end_offset),
                              Position(offset=node.end_position.offset,
                                       line=node.end_position.line,
                                       col=node.end_position.col),
                              path=path)


AnyNode = Union[VirtualNode, bblfsh.Node]
