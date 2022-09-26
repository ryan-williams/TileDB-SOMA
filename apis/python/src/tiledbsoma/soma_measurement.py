from typing import Any, Dict, Optional

import tiledb

from .soma_collection import SOMACollection
from .soma_dataframe import SOMADataFrame
from .tiledb_platform_config import TileDBPlatformConfig


class SOMAMeasurement(SOMACollection):
    """
    A ``SOMAMeasurement`` is a sub-element of a ``SOMAExperiment``, and is otherwise a specialized ``SOMACollection`` with pre-defined fields:

    ``var``: ``SOMADataFrame``

    Primary annotations on the variable axis, for variables in this measurement (i.e., annotates columns of ``X``). The contents of the ``soma_rowid`` pseudo-column define the variable index domain, AKA varid. All variables for this measurement must be defined in this dataframe.

    ``X``: ``SOMACollection`` of ``SOMASparseNdArray``

    A collection of sparse matrices, each containing measured feature values. Each matrix is indexed by ``[obsid, varid]``.

    ``obsm``: ``SOMACollection`` of ``SOMADenseNdArray``

    A collection of dense matrices containing annotations of each ``obs`` row. Has the same shape as ``obs``, and is indexed with ``obsid``.

    ``obsp``: ``SOMACollection`` of ``SOMASparseNdArray``

    A collection of sparse matrices containing pairwise annotations of each ``obs`` row. Indexed with ``[obsid_1, obsid_2]``.

    ``varm``: ``SOMACollection`` of ``SOMADenseNdArray``

    A collection of dense matrices containing annotations of each ``var`` row. Has the same shape as ``var``, and is indexed with ``varid``.

    ``varp``: ``SOMACollection`` of ``SOMASparseNdArray``

    A collection of sparse matrices containing pairwise annotations of each ``var`` row. Indexed with ``[varid_1, varid_2]``
    """

    _constructors: Dict[str, Any]
    _cached_members: Dict[str, Any]

    def __init__(
        self,
        uri: str,
        *,
        name: Optional[str] = None,
        # Non-top-level objects can have a parent to propagate context, depth, etc.
        parent: Optional[SOMACollection] = None,
        # Top-level objects should specify these:
        tiledb_platform_config: Optional[TileDBPlatformConfig] = None,
        ctx: Optional[tiledb.Ctx] = None,
    ):
        """
        Also see the ``TileDBObject`` constructor.
        """
        super().__init__(
            uri=uri,
            name=name,
            parent=parent,
            tiledb_platform_config=tiledb_platform_config,
            ctx=ctx,
        )
        self._constructors = {
            "var": SOMADataFrame,
            "X": SOMACollection,
            "obsm": SOMACollection,
            "obsp": SOMACollection,
            "varm": SOMACollection,
            "varp": SOMACollection,
        }
        self._cached_members = {}

    def create(self) -> None:
        """
        Creates the data structure on disk/S3/cloud.
        """
        super().create()

    def __getattr__(self, name: str) -> Any:
        """
        Implements ``experiment.var``, ``experiment.X``, etc.
        """
        if name in self._constructors:
            if name not in self._cached_members:
                child_uri = self._get_child_uri(name)
                self._cached_members[name] = self._constructors[name](
                    uri=child_uri, name=name, parent=self
                )
            return self._cached_members[name]
        else:
            # Unlike __getattribute__ this is _only_ called when the member isn't otherwise
            # resolvable. So raising here is the right thing to do.
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def constrain(self) -> None:
        """
        Checks constraints on the collection. Raises an exception if any is violated.
        """
        # TODO: find a good spot to call this from.

        # TODO: resolve polymorphism issues
        # for attr in ["obsp", "varp", "X"]:
        #    if attr in self:
        #        # error: "TileDBObject" has no attribute "__iter__" (not iterable)  [attr-defined]
        #        for element in self[attr]:
        #            # TODO: make this a SOMACollection method
        #            if not isinstance(element, SOMASparseNdArray):
        #                raise Exception(
        #                    f"element {element.name} of {self.type}.{attr} should be SOMASparseNdArray; got {element.__class__.__name__}"
        #                )

        # for attr in ["obsm", "varm"]:
        #    if attr in self:
        #        # error: "TileDBObject" has no attribute "__iter__" (not iterable)  [attr-defined]
        #        for element in self[attr]:
        #            # TODO: make this a SOMACollection method
        #            if not isinstance(element, SOMADenseNdArray):
        #                raise Exception(
        #                    f"element {element.name} of {self.type}.{attr} should be SOMADenseNdArray; got {element.__class__.__name__}"
        #                )

    # ``X`` collection values
    # o All matrices must have the shape ``(#obs, #var)``.
    # o The domain of the first dimension is the values of ``obs.soma_rowid``, and the index domain of
    #   the second dimension is the values of ``var.soma_rowid`` in the containing ``SOMAMeasurement``.

    # ``obsm`` collection values
    # o All matrices must have the shape ``(#obs, M)``, where ``M`` is user-defined.
    # o The domain of the first dimension is the values of ``obs.soma_rowid``.

    # ``obsp`` collection values
    # o All matrices must have the shape ``(#obs, #obs)``.
    # o The domain of both dimensions is the values of ``obs.soma_rowid``.

    # ``varm`` collection values
    # o All matrices must have the shape ``(#var, M)``, where ``M`` is user-defined.
    # o The domain of the first dimension is the values of ``var.soma_rowid``.

    # ``varp`` collection values
    # o All matrices must have the shape ``(#var, #var)``.
    # o The domain of both dimensions is the values of ``var.soma_rowid``.