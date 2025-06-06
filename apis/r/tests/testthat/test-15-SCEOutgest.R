test_that("Load SCE object from ExperimentQuery mechanics", {
  skip_if(!extended_tests() || covr_tests())
  skip_if_not_installed("SingleCellExperiment", .MINIMUM_SCE_VERSION("c"))
  uri <- tempfile(pattern = "sce-experiment-query-whole")

  n_obs <- 20L
  n_var <- 10L
  n_pcs <- 50L
  n_ics <- 30L
  n_umaps <- 2L
  experiment <- create_and_populate_experiment(
    uri = uri,
    n_obs = n_obs,
    n_var = n_var,
    X_layer_names = c("counts", "logcounts"),
    obsm_layers = c(X_pca = n_pcs, "dense:X_ica" = n_ics, X_umap = n_umaps),
    # No varm in SingleCellExperiment
    obsp_layer_names = "connectivities",
    varp_layer_names = "network",
    mode = "READ"
  )
  on.exit(experiment$close())

  # Create the query
  query <- SOMAExperimentAxisQuery$new(
    experiment = experiment,
    measurement_name = "RNA"
  )
  expect_warning(obj <- query$to_single_cell_experiment())
  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var, n_obs))
  expect_identical(
    rownames(obj),
    paste0("var", query$var_joinids()$as_vector())
  )
  expect_identical(
    colnames(obj),
    paste0("obs", query$obs_joinids()$as_vector())
  )
  expect_true(all(
    query$obs_df$attrnames() %in% names(SingleCellExperiment::colData(obj))
  ))
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  for (slot in SummarizedExperiment::assayNames(obj)) {
    expect_s4_class(mat <- SummarizedExperiment::assay(obj, slot), "dgTMatrix")
    expect_identical(rownames(mat), rownames(obj))
    expect_identical(colnames(mat), colnames(obj))
  }
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("ICA", "PCA", "UMAP")
  )
  expect_is(pca <- SingleCellExperiment::reducedDim(obj, "PCA"), "matrix")
  expect_identical(dim(pca), c(n_obs, n_pcs))
  expect_identical(rownames(pca), colnames(obj))
  expect_identical(colnames(pca), paste0("PC", seq_len(n_pcs)))
  expect_is(ica <- SingleCellExperiment::reducedDim(obj, "ICA"), "matrix")
  expect_identical(dim(ica), c(n_obs, n_ics))
  expect_identical(rownames(ica), colnames(obj))
  expect_identical(colnames(ica), paste0("IC", seq_len(n_ics)))
  expect_is(umap <- SingleCellExperiment::reducedDim(obj, "UMAP"), "matrix")
  expect_identical(dim(umap), c(n_obs, n_umaps))
  expect_identical(rownames(umap), colnames(obj))
  expect_identical(colnames(umap), paste0("UMAP", seq_len(n_umaps)))
  expect_identical(SingleCellExperiment::colPairNames(obj), "connectivities")
  expect_s4_class(SingleCellExperiment::colPair(obj, "connectivities"), "SelfHits")
  expect_s4_class(
    graph <- SingleCellExperiment::colPair(obj, "connectivities", asSparse = TRUE),
    "dgCMatrix"
  )
  expect_identical(dim(graph), c(n_obs, n_obs))
  expect_identical(SingleCellExperiment::rowPairNames(obj), "network")
  expect_s4_class(SingleCellExperiment::rowPair(obj, "network"), "SelfHits")
  expect_s4_class(
    net <- SingleCellExperiment::rowPair(obj, "network", asSparse = TRUE),
    "dgCMatrix"
  )
  expect_identical(dim(net), c(n_var, n_var))
  # Test named
  expect_warning(obj <- query$to_single_cell_experiment(
    obs_index = "string_column",
    var_index = "quux"
  ))
  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var, n_obs))
  expect_identical(
    rownames(obj),
    query$var("quux")$concat()$GetColumnByName("quux")$as_vector()
  )
  expect_identical(
    colnames(obj),
    query$obs("string_column")$concat()$GetColumnByName("string_column")$as_vector()
  )
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  expect_false(all(
    query$obs_df$attrnames() %in% names(SingleCellExperiment::colData(obj))
  ))
  expect_true(all(
    setdiff(query$obs_df$attrnames(), "string_column") %in% names(SingleCellExperiment::colData(obj))
  ))
  for (slot in SummarizedExperiment::assayNames(obj)) {
    expect_s4_class(mat <- SummarizedExperiment::assay(obj, slot), "dgTMatrix")
    expect_identical(rownames(mat), rownames(obj))
    expect_identical(colnames(mat), colnames(obj))
  }
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("ICA", "PCA", "UMAP")
  )
  for (rd in SingleCellExperiment::reducedDimNames(obj)) {
    expect_is(mat <- SingleCellExperiment::reducedDim(obj, rd), "matrix")
    expect_identical(nrow(mat), n_obs)
    expect_identical(rownames(mat), colnames(obj))
  }
  # Test `X_layers`
  expect_warning(obj <- query$to_single_cell_experiment("counts"))
  expect_identical(SummarizedExperiment::assayNames(obj), "counts")
  expect_s4_class(SummarizedExperiment::assay(obj, "counts"), "dgTMatrix")
  expect_error(SummarizedExperiment::assay(obj, "logcounts"))
  expect_warning(obj <- query$to_single_cell_experiment("logcounts"))
  expect_identical(SummarizedExperiment::assayNames(obj), "logcounts")
  expect_s4_class(SummarizedExperiment::assay(obj, "logcounts"), "dgTMatrix")
  expect_error(SummarizedExperiment::assay(obj, "counts"))
  expect_warning(obj <- query$to_single_cell_experiment(c(matrix = "logcounts")))
  expect_identical(SummarizedExperiment::assayNames(obj), "matrix")
  expect_s4_class(SummarizedExperiment::assay(obj, "matrix"), "dgTMatrix")
  # Test suppress reductions
  expect_no_condition(obj <- query$to_single_cell_experiment(obsm_layers = FALSE))
  expect_length(SingleCellExperiment::reducedDimNames(obj), 0L)
  expect_no_condition(obj <- query$to_single_cell_experiment(obsm_layers = NA))
  expect_length(SingleCellExperiment::reducedDimNames(obj), 0L)
  expect_no_condition(obj <- query$to_single_cell_experiment(obsm_layers = c(UMAP = "X_umap")))
  expect_identical(SingleCellExperiment::reducedDimNames(obj), "UMAP")
  expect_error(SingleCellExperiment::reducedDim(obj, "PCA"))
  # # Test suppress graphs
  expect_no_condition(
    obj <- query$to_single_cell_experiment(obsm_layers = FALSE, obsp_layers = FALSE)
  )
  expect_length(SingleCellExperiment::colPairNames(obj), 0L)
  # Test suppress cell-level meta data
  expect_no_condition(
    obj <- query$to_single_cell_experiment(obsm_layers = FALSE, obs_column_names = FALSE)
  )
  expect_false(any(
    query$obs_df$attrnames() %in% names(SingleCellExperiment::colData(obj))
  ))
  # Test `X_layers` assertions
  expect_error(query$to_single_cell_experiment(FALSE))
  expect_error(query$to_single_cell_experiment(1))
  expect_error(query$to_single_cell_experiment(list("counts", "logcounts")))
  expect_error(query$to_single_cell_experiment(c(counts = "tomato")))
  # Test `obs_index` assertions
  expect_error(query$to_single_cell_experiment(obs_index = FALSE))
  expect_error(query$to_single_cell_experiment(obs_index = NA_character_))
  expect_error(query$to_single_cell_experiment(obs_index = 1))
  expect_error(query$to_single_cell_experiment(obs_index = c("string_column", "int_column")))
  expect_error(query$to_single_cell_experiment(obs_index = "tomato"))
  # Test `var_index` assertions
  expect_error(query$to_single_cell_experiment(var_index = FALSE))
  expect_error(query$to_single_cell_experiment(var_index = NA_character_))
  expect_error(query$to_single_cell_experiment(var_index = 1))
  expect_error(query$to_single_cell_experiment(var_index = c("quux", "xyzzy")))
  expect_error(query$to_single_cell_experiment(var_index = "tomato"))
  # Test `obs_column_names` assertions
  expect_error(query$to_single_cell_experiment(obs_column_names = 1L))
  expect_error(query$to_single_cell_experiment(obs_column_names = c(
    NA_character_,
    NA_character_
  )))
  expect_error(query$to_single_cell_experiment(obs_column_names = c(TRUE, FALSE)))
  expect_error(query$to_single_cell_experiment(obs_column_names = "tomato"))
  # Test `var_column_names` assertions
  expect_error(query$to_single_cell_experiment(var_column_names = 1L))
  expect_error(query$to_single_cell_experiment(var_column_names = c(
    NA_character_,
    NA_character_
  )))
  expect_error(query$to_single_cell_experiment(var_column_names = c(TRUE, FALSE)))
  expect_error(query$to_single_cell_experiment(var_column_names = "tomato"))
  # Test `obsm_layers` assertions
  expect_error(query$to_single_cell_experiment(obsm_layers = 1L))
  expect_error(query$to_single_cell_experiment(obsm_layers = "tomato"))
  # Test `obsp_layers` assertions
  expect_error(query$to_single_cell_experiment(obsp_layers = 1L))
  expect_error(query$to_single_cell_experiment(obsm_layers = FALSE, obsp_layers = "tomato"))
  # Test `varp_layers` assertions
  expect_error(query$to_single_cell_experiment(obsp_layers = 1L))
  expect_error(query$to_single_cell_experiment(obsm_layers = FALSE, obsp_layers = "tomato"))
})

test_that("Load SCE object with dropped levels", {
  skip_if(!extended_tests() || covr_tests())
  skip_if_not_installed("SingleCellExperiment", .MINIMUM_SCE_VERSION("c"))
  uri <- tempfile(pattern = "sce-experiment-query-drop")

  n_obs <- 20L
  n_var <- 10L
  experiment <- create_and_populate_experiment(
    uri = uri,
    n_obs = n_obs,
    n_var = n_var,
    X_layer_names = c("counts", "logcounts"),
    factors = TRUE,
    mode = "READ"
  )
  on.exit(experiment$close(), add = TRUE, after = FALSE)

  # Create the query
  query <- SOMAExperimentAxisQuery$new(
    experiment = experiment,
    measurement_name = "RNA",
    obs_query = SOMAAxisQuery$new(coords = seq.int(to = floor(n_obs / 3))),
    var_query = SOMAAxisQuery$new(coords = seq.int(to = floor(n_var / 3)))
  )

  # Expect both levels to be present in `grp`, even though only one value is
  expect_s4_class(sce <- query$to_single_cell_experiment(), "SingleCellExperiment")
  expect_in("grp", names(SingleCellExperiment::colData(sce)))
  expect_s3_class(obs <- SingleCellExperiment::colData(sce)$grp, "factor")
  expect_identical(levels(obs), c("lvl1", "lvl2"))
  expect_identical(unique(as.vector(obs)), "lvl1")
  expect_in("grp", names(SingleCellExperiment::rowData(sce)))
  expect_s3_class(var <- SingleCellExperiment::rowData(sce)$grp, "factor")
  expect_identical(levels(var), c("lvl1", "lvl2"))
  expect_identical(unique(as.vector(var)), "lvl1")

  # Do the same, but drop levels
  expect_s4_class(
    dropped <- query$to_single_cell_experiment(drop_levels = TRUE),
    "SingleCellExperiment"
  )
  expect_in("grp", names(SingleCellExperiment::colData(dropped)))
  expect_s3_class(obsd <- SingleCellExperiment::colData(dropped)$grp, "factor")
  expect_identical(levels(obsd), "lvl1")
  expect_identical(unique(as.vector(obsd)), "lvl1")
  expect_in("grp", names(SingleCellExperiment::rowData(dropped)))
  expect_s3_class(vard <- SingleCellExperiment::rowData(dropped)$grp, "factor")
  expect_identical(levels(vard), "lvl1")
  expect_identical(unique(as.vector(vard)), "lvl1")

  # Test assertions
  expect_error(query$to_single_cell_experiment(drop_levels = NA))
  expect_error(query$to_single_cell_experiment(drop_levels = 1L))
  expect_error(query$to_single_cell_experiment(drop_levels = "drop"))
  expect_error(query$to_single_cell_experiment(drop_levels = c(TRUE, TRUE)))
})

test_that("Load SCE object from sliced ExperimentQuery", {
  skip_if(!extended_tests() || covr_tests())
  skip_if_not_installed("SingleCellExperiment", .MINIMUM_SCE_VERSION("c"))
  uri <- tempfile(pattern = "sce-experiment-query-sliced")
  n_obs <- 1001L
  n_var <- 99L
  n_pcs <- 50L
  n_umaps <- 2L
  experiment <- create_and_populate_experiment(
    uri = uri,
    n_obs = n_obs,
    n_var = n_var,
    X_layer_names = c("counts", "logcounts"),
    obsm_layers = c(X_pca = n_pcs, X_umap = n_umaps),
    # No varm in SingleCellExperiment
    obsp_layer_names = "connectivities",
    varp_layer_names = "network",
    mode = "READ"
  )
  on.exit(experiment$close())
  # Create the query
  obs_slice <- bit64::as.integer64(seq(3, 72))
  var_slice <- bit64::as.integer64(seq(7, 21))
  query <- SOMAExperimentAxisQuery$new(
    experiment = experiment,
    measurement_name = "RNA",
    obs_query = SOMAAxisQuery$new(coords = list(soma_joinid = obs_slice)),
    var_query = SOMAAxisQuery$new(coords = list(soma_joinid = var_slice))
  )
  n_var_slice <- length(var_slice)
  n_obs_slice <- length(obs_slice)
  expect_no_condition(obj <- query$to_single_cell_experiment())
  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var_slice, n_obs_slice))
  expect_identical(
    rownames(obj),
    paste0("var", query$var_joinids()$as_vector())
  )
  expect_identical(
    colnames(obj),
    paste0("obs", query$obs_joinids()$as_vector())
  )
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("PCA", "UMAP")
  )
  expect_identical(SingleCellExperiment::colPairNames(obj), "connectivities")
  expect_identical(SingleCellExperiment::rowPairNames(obj), "network")
  # Test named
  expect_no_condition(obj <- query$to_single_cell_experiment(obs_index = "string_column", var_index = "quux"))
  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var_slice, n_obs_slice))
  expect_identical(
    rownames(obj),
    query$var("quux")$concat()$GetColumnByName("quux")$as_vector()
  )
  expect_identical(
    colnames(obj),
    query$obs("string_column")$concat()$GetColumnByName("string_column")$as_vector()
  )
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("PCA", "UMAP")
  )
  expect_identical(SingleCellExperiment::colPairNames(obj), "connectivities")
  expect_identical(SingleCellExperiment::rowPairNames(obj), "network")
})

test_that("Load SCE object from indexed ExperimentQuery", {
  skip_if(!extended_tests() || covr_tests())
  skip_if_not_installed("SingleCellExperiment", .MINIMUM_SCE_VERSION("c"))
  uri <- tempfile(pattern = "sce-experiment-query-value-filters")

  n_obs <- 1001L
  n_var <- 99L
  n_pcs <- 50L
  n_umaps <- 2L
  obs_label_values <- c("1003", "1007", "1038", "1099")
  var_label_values <- c("1018", "1034", "1067")
  experiment <- create_and_populate_experiment(
    uri = uri,
    n_obs = n_obs,
    n_var = n_var,
    X_layer_names = c("counts", "logcounts"),
    obsm_layers = c(X_pca = n_pcs, X_umap = n_umaps),
    # No varm in SingleCellExperiment
    obsp_layer_names = "connectivities",
    varp_layer_names = "network",
    mode = "READ"
  )
  on.exit(experiment$close())
  # Create the query
  obs_value_filter <- paste0(
    sprintf("string_column == '%s'", obs_label_values),
    collapse = "||"
  )
  var_value_filter <- paste0(
    sprintf("quux == '%s'", var_label_values),
    collapse = "||"
  )
  query <- SOMAExperimentAxisQuery$new(
    experiment = experiment,
    measurement_name = "RNA",
    obs_query = SOMAAxisQuery$new(value_filter = obs_value_filter),
    var_query = SOMAAxisQuery$new(value_filter = var_value_filter)
  )
  n_var_select <- length(var_label_values)
  n_obs_select <- length(obs_label_values)
  expect_no_condition(obj <- query$to_single_cell_experiment())

  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var_select, n_obs_select))
  expect_identical(
    rownames(obj),
    paste0("var", query$var_joinids()$as_vector())
  )
  expect_identical(
    colnames(obj),
    paste0("obs", query$obs_joinids()$as_vector())
  )
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("PCA", "UMAP")
  )
  expect_identical(SingleCellExperiment::colPairNames(obj), "connectivities")
  expect_identical(SingleCellExperiment::rowPairNames(obj), "network")
  # Test named
  expect_no_condition(
    obj <- query$to_single_cell_experiment(obs_index = "string_column", var_index = "quux")
  )
  expect_s4_class(obj, "SingleCellExperiment")
  expect_identical(dim(obj), c(n_var_select, n_obs_select))
  expect_identical(
    rownames(obj),
    query$var("quux")$concat()$GetColumnByName("quux")$as_vector()
  )
  expect_identical(
    colnames(obj),
    query$obs("string_column")$concat()$GetColumnByName("string_column")$as_vector()
  )
  expect_identical(SingleCellExperiment::mainExpName(obj), "RNA")
  expect_identical(
    sort(SummarizedExperiment::assayNames(obj)),
    c("counts", "logcounts")
  )
  expect_identical(
    sort(SingleCellExperiment::reducedDimNames(obj)),
    c("PCA", "UMAP")
  )
  expect_identical(SingleCellExperiment::colPairNames(obj), "connectivities")
  expect_identical(SingleCellExperiment::rowPairNames(obj), "network")
})
