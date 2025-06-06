% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SOMAExperimentAxisQuery.R
\name{SOMAExperimentAxisQuery}
\alias{SOMAExperimentAxisQuery}
\title{\code{SOMAExperiment} Axis Query}
\description{
Perform an axis-based query against a \code{\link{SOMAExperiment}}.

\code{SOMAExperimentAxisQuery} allows easy selection and extraction of data from a
single \code{\link{SOMAMeasurement}} in a \code{\link{SOMAExperiment}}, by \code{obs}/\code{var} (axis)
coordinates and/or value filter. The primary use for this class is slicing
\code{\link{SOMAExperiment}} \code{X} layers by \code{obs} or \code{var} value and/or coordinates.
(lifecycle: maturing)
\subsection{X Layer Support}{

Slicing on \code{\link{SOMASparseNDArray}} \code{X} matrices is supported;
slicing on \code{\link{SOMADenseNDArray}} is not supported at this time.
}

\subsection{Result Size}{

\code{SOMAExperimentAxisQuery} query class assumes it can store the full result of
both axis dataframe queries in memory, and only provides incremental access
to the underlying X NDArray. Accessors such as \code{n_obs} and \code{n_vars} codify
this in the class.
}
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{experiment}}{The parent \code{\link{SOMAExperiment}} object.}

\item{\code{indexer}}{The \code{\link{SOMAAxisIndexer}} object.}

\item{\code{obs_query}}{The \code{obs} \code{\link{SOMAAxisQuery}} object.}

\item{\code{var_query}}{The \code{var} \code{\link{SOMAAxisQuery}} object.}

\item{\code{n_obs}}{The number of \code{obs} axis query results.}

\item{\code{n_vars}}{The number of \code{var} axis query results.}

\item{\code{obs_df}}{The \code{obs} \code{\link{SOMADataFrame}} object.}

\item{\code{var_df}}{The \code{var} \code{\link{SOMADataFrame}} object for the specified
\code{measurement_name}.}

\item{\code{ms}}{The \code{\link{SOMAMeasurement}} object for the specified
\code{measurement_name}.}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-SOMAExperimentAxisQuery-new}{\code{SOMAExperimentAxisQuery$new()}}
\item \href{#method-SOMAExperimentAxisQuery-obs}{\code{SOMAExperimentAxisQuery$obs()}}
\item \href{#method-SOMAExperimentAxisQuery-var}{\code{SOMAExperimentAxisQuery$var()}}
\item \href{#method-SOMAExperimentAxisQuery-obs_joinids}{\code{SOMAExperimentAxisQuery$obs_joinids()}}
\item \href{#method-SOMAExperimentAxisQuery-var_joinids}{\code{SOMAExperimentAxisQuery$var_joinids()}}
\item \href{#method-SOMAExperimentAxisQuery-X}{\code{SOMAExperimentAxisQuery$X()}}
\item \href{#method-SOMAExperimentAxisQuery-obsm}{\code{SOMAExperimentAxisQuery$obsm()}}
\item \href{#method-SOMAExperimentAxisQuery-varm}{\code{SOMAExperimentAxisQuery$varm()}}
\item \href{#method-SOMAExperimentAxisQuery-obsp}{\code{SOMAExperimentAxisQuery$obsp()}}
\item \href{#method-SOMAExperimentAxisQuery-varp}{\code{SOMAExperimentAxisQuery$varp()}}
\item \href{#method-SOMAExperimentAxisQuery-read}{\code{SOMAExperimentAxisQuery$read()}}
\item \href{#method-SOMAExperimentAxisQuery-to_sparse_matrix}{\code{SOMAExperimentAxisQuery$to_sparse_matrix()}}
\item \href{#method-SOMAExperimentAxisQuery-to_seurat}{\code{SOMAExperimentAxisQuery$to_seurat()}}
\item \href{#method-SOMAExperimentAxisQuery-to_seurat_assay}{\code{SOMAExperimentAxisQuery$to_seurat_assay()}}
\item \href{#method-SOMAExperimentAxisQuery-to_seurat_reduction}{\code{SOMAExperimentAxisQuery$to_seurat_reduction()}}
\item \href{#method-SOMAExperimentAxisQuery-to_seurat_graph}{\code{SOMAExperimentAxisQuery$to_seurat_graph()}}
\item \href{#method-SOMAExperimentAxisQuery-to_single_cell_experiment}{\code{SOMAExperimentAxisQuery$to_single_cell_experiment()}}
\item \href{#method-SOMAExperimentAxisQuery-clone}{\code{SOMAExperimentAxisQuery$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-new"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-new}{}}}
\subsection{Method \code{new()}}{
Create a new \code{SOMAExperimentAxisQuery} object.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$new(
  experiment,
  measurement_name,
  obs_query = NULL,
  var_query = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{experiment}}{A \code{\link{SOMAExperiment}} object.}

\item{\code{measurement_name}}{The name of the measurement to query.}

\item{\code{obs_query, var_query}}{An \code{\link{SOMAAxisQuery}} object for the obs/var
axis.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-obs"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-obs}{}}}
\subsection{Method \code{obs()}}{
Retrieve obs \link{TableReadIter}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$obs(column_names = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{column_names}}{A character vector of column names to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-var"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-var}{}}}
\subsection{Method \code{var()}}{
Retrieve var \code{\link[arrow:Table-class]{arrow::Table}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$var(column_names = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{column_names}}{A character vector of column names to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-obs_joinids"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-obs_joinids}{}}}
\subsection{Method \code{obs_joinids()}}{
Retrieve \code{soma_joinids} as an \code{\link[arrow:array-class]{arrow::Array}} for \code{obs}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$obs_joinids()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-var_joinids"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-var_joinids}{}}}
\subsection{Method \code{var_joinids()}}{
Retrieve \code{soma_joinids} as an \code{\link[arrow:array-class]{arrow::Array}} for \code{var}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$var_joinids()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-X"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-X}{}}}
\subsection{Method \code{X()}}{
Retrieves an \code{X} layer as a \link{SOMASparseNDArrayRead}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$X(layer_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{layer_name}}{The name of the layer to retrieve.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-obsm"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-obsm}{}}}
\subsection{Method \code{obsm()}}{
Retrieves an \code{obsm} layer as a \code{\link{SOMASparseNDArrayRead}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$obsm(layer_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{layer_name}}{The name of the layer to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-varm"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-varm}{}}}
\subsection{Method \code{varm()}}{
Retrieves a \code{varm} layer as a \code{\link{SOMASparseNDArrayRead}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$varm(layer_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{layer_name}}{The name of the layer to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-obsp"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-obsp}{}}}
\subsection{Method \code{obsp()}}{
Retrieves an \code{obsp} layer as a \code{\link{SOMASparseNDArrayRead}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$obsp(layer_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{layer_name}}{The name of the layer to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-varp"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-varp}{}}}
\subsection{Method \code{varp()}}{
Retrieves a \code{varp} layer as a \code{\link{SOMASparseNDArrayRead}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$varp(layer_name)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{layer_name}}{The name of the layer to retrieve}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-read"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-read}{}}}
\subsection{Method \code{read()}}{
Reads the entire query result as a list of
\code{\link[arrow:Table-class]{arrow::Table}}s. This is a low-level routine intended to be used by
loaders for other in-core formats, such as \code{Seurat}, which can be created
from the resulting Tables.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$read(
  X_layers = NULL,
  obs_column_names = NULL,
  var_column_names = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_layers}}{The name(s) of the \code{X} layer(s) to read and return.}

\item{\code{obs_column_names, var_column_names}}{Specify which column names in
\code{var} and \code{obs} dataframes to read and return.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_sparse_matrix"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_sparse_matrix}{}}}
\subsection{Method \code{to_sparse_matrix()}}{
Retrieve a collection layer as a sparse matrix with named
dimensions.

Load any layer from the \code{X}, \code{obsm}, \code{varm}, \code{obsp}, or \code{varp}
collections as a \link[Matrix:sparseMatrix-class]{sparse matrix}.

By default the matrix dimensions are named using the \code{soma_joinid} values
in the specified layer's dimensions (e.g., \code{soma_dim_0}). However,
dimensions can be named using values from any \code{obs} or \code{var} column that
uniquely identifies each record by specifying the \code{obs_index} and
\code{var_index} arguments.

For layers in \code{obsm} or \code{varm}, the column axis (the axis not
indexed by \dQuote{\code{obs}} or \dQuote{\code{var}}) is set to the
range of values present in \dQuote{\code{soma_dim_1}}; this ensures
that gaps in this axis are preserved (eg. when a query for
\dQuote{\code{obs}} that results in selecting entries that are all zero
for a given PC)
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_sparse_matrix(
  collection,
  layer_name,
  obs_index = NULL,
  var_index = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{collection}}{The \code{\link{SOMACollection}} containing the layer of
interest, either: \code{"X"}, \code{"obsm"}, \code{"varm"}, \code{"obsp"}, or \code{"varp"}.}

\item{\code{layer_name}}{Name of the layer to retrieve from the \code{collection}.}

\item{\code{obs_index, var_index}}{Name of the column in \code{obs} or \code{var}
(\code{var_index}) containing values that should be used as dimension labels
in the resulting matrix. Whether the values are used as row or column
labels depends on the selected \code{collection}:\tabular{rll}{
   Collection \tab \code{obs_index} \tab \code{var_index} \cr
   \code{X} \tab row names \tab column names \cr
   \code{obsm} \tab row names \tab ignored \cr
   \code{varm} \tab ignored \tab row names \cr
   \code{obsp} \tab row and column names \tab ignored \cr
   \code{varp} \tab ignored \tab row and column names \cr
}}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A \code{\link[Matrix:sparseMatrix-class]{Matrix::sparseMatrix}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_seurat"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_seurat}{}}}
\subsection{Method \code{to_seurat()}}{
Loads the query as a \code{\link[SeuratObject]{Seurat}} object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_seurat(
  X_layers = c(counts = "counts", data = "logcounts"),
  obs_index = NULL,
  var_index = NULL,
  obs_column_names = NULL,
  var_column_names = NULL,
  obsm_layers = NULL,
  varm_layers = NULL,
  obsp_layers = NULL,
  drop_levels = FALSE,
  version = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_xlayers()}}

\item{\code{obs_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index()}}

\item{\code{var_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index(axis = 'var')}}

\item{\code{obs_column_names}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_metadata_names()}}

\item{\code{var_column_names}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_metadata_names(axis = 'var')}}

\item{\code{obsm_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_mlayers()}}

\item{\code{varm_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_mlayers(axis = 'varm')}}

\item{\code{obsp_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_players()}}

\item{\code{drop_levels}}{Drop unused levels from \code{obs} and \code{var} factor columns}

\item{\code{version}}{Assay version to read query in as; by default, will try to
infer assay type from the measurement itself}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A \code{\link[SeuratObject]{Seurat}} object
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_seurat_assay"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_seurat_assay}{}}}
\subsection{Method \code{to_seurat_assay()}}{
Loads the query as a Seurat \code{\link[SeuratObject]{Assay}}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_seurat_assay(
  X_layers = c(counts = "counts", data = "logcounts"),
  obs_index = NULL,
  var_index = NULL,
  var_column_names = NULL,
  drop_levels = FALSE,
  version = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_xlayers()}}

\item{\code{obs_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index()}}

\item{\code{var_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index(axis = 'var')}}

\item{\code{var_column_names}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_metadata_names(axis = 'var')}}

\item{\code{drop_levels}}{Drop unused levels from \code{var} factor columns}

\item{\code{version}}{Assay version to read query in as; by default, will try to
infer assay type from the measurement itself}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
An \code{\link[SeuratObject]{Assay}} object
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_seurat_reduction"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_seurat_reduction}{}}}
\subsection{Method \code{to_seurat_reduction()}}{
Loads the query as a Seurat
\link[SeuratObject:DimReduc]{dimensional reduction}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_seurat_reduction(
  obsm_layer,
  varm_layer = NULL,
  obs_index = NULL,
  var_index = NULL
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{obsm_layer}}{Name of array in \code{obsm} to load as the
cell embeddings}

\item{\code{varm_layer}}{Name of the array in \code{varm} to load as the
feature loadings; by default, will try to determine \code{varm_layer}
from \code{obsm_layer}}

\item{\code{obs_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index()}}

\item{\code{var_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index(axis = 'var')}}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A \code{\link[SeuratObject]{DimReduc}} object
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_seurat_graph"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_seurat_graph}{}}}
\subsection{Method \code{to_seurat_graph()}}{
Loads the query as a Seurat \link[SeuratObject:Graph]{graph}
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_seurat_graph(obsp_layer, obs_index = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{obsp_layer}}{Name of array in \code{obsp} to load as the graph}

\item{\code{obs_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index()}}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A \code{\link[SeuratObject]{Graph}} object
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-to_single_cell_experiment"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-to_single_cell_experiment}{}}}
\subsection{Method \code{to_single_cell_experiment()}}{
Loads the query as a
\code{\link[SingleCellExperiment]{SingleCellExperiment}} object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$to_single_cell_experiment(
  X_layers = NULL,
  obs_index = NULL,
  var_index = NULL,
  obs_column_names = NULL,
  var_column_names = NULL,
  obsm_layers = NULL,
  obsp_layers = NULL,
  varp_layers = NULL,
  drop_levels = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_xlayers('sce')}}

\item{\code{obs_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index('sce')}}

\item{\code{var_index}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_index('sce', 'var')}}

\item{\code{obs_column_names}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_metadata_names('sce')}}

\item{\code{var_column_names}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_metadata_names('sce', 'var')}}

\item{\code{obsm_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_mlayers('sce')}}

\item{\code{obsp_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_players('sce')}}

\item{\code{varp_layers}}{\Sexpr[results=rd]{tiledbsoma:::rd_outgest_players('sce', 'varp')}}

\item{\code{drop_levels}}{Drop unused levels from \code{obs} and \code{var} factor columns}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A \code{\link[SingleCellExperiment]{SingleCellExperiment}} object
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-SOMAExperimentAxisQuery-clone"></a>}}
\if{latex}{\out{\hypertarget{method-SOMAExperimentAxisQuery-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{SOMAExperimentAxisQuery$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
