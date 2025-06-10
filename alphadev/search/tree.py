
class NodeBase:

class TreeBase:
    """
    Base class for tree structures.
    """

    def reset(self, last_action: Optional[int] = None):
    def append_child(self, parent: NodeBase, action: int) -> Optional[NodeBase]:
    def get_node(self, index) -> NodeBase:
    def get_root(self):
    def get_by_offset(self, offset: int) -> NodeBase:
    def is_full(self) -> bool: