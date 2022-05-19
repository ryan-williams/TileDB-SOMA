import tiledb
from .soma_options import SOMAOptions
from .tiledb_object import TileDBObject
from .tiledb_group import TileDBGroup

from typing import Optional, List


class TileDBArray(TileDBObject):
    """
    Wraps arrays from TileDB-Py by retaining a URI, verbose flag, etc.
    """

    def __init__(
        self,
        uri: str,
        name: str,
        parent: Optional[TileDBGroup] = None,
    ):
        """
        See the TileDBObject constructor.
        """
        super().__init__(uri=uri, name=name, parent=parent)

    def object_type(self):
        """
        This should be implemented by child classes and should return what tiledb.object_type(uri)
        returns for objects of a given type -- nominally 'group' or 'array'.
        """
        return "array"

    def open(self):
        """
        Returns the TileDB array. The caller should do A.close() on the return value after finishing
        with it.
        """
        A = tiledb.open(self.uri)
        return A

    def schema(self):
        """
        Returns the TileDB array schema.
        """
        with tiledb.open(self.uri) as A:
            return A.schema

    def get_dim_names(self) -> List[str]:
        """
        Reads the dimension names from the schema: for example, ['obs_id', 'var_id'].
        """
        with tiledb.open(self.uri) as A:
            return [A.schema.domain.dim(i).name for i in range(A.schema.domain.ndim)]

    def get_attr_names(self) -> List[str]:
        """
        Reads the attribute names from the schema: for example, the list of column names in a dataframe.
        """
        with tiledb.open(self.uri) as A:
            return [A.schema.attr(i).name for i in range(A.schema.nattr)]

    def has_attr_name(self, attr_name: str) -> bool:
        """
        Returns true if the array has the specified attribute name, false otherwise.
        """
        return attr_name in self.get_attr_names()