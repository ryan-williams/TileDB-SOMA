test_that("Write Assay mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", .MINIMUM_SEURAT_VERSION("c"))

  uri <- tempfile(pattern = "write-assay")
  collection <- SOMACollectionCreate(uri)
  on.exit(collection$close(), add = TRUE, after = FALSE)

  rna <- get_data("pbmc_small", package = "SeuratObject")[["RNA"]]
  expect_no_condition(ms <- write_soma(rna, soma_parent = collection))
  on.exit(ms$close(), add = TRUE, after = FALSE)
  expect_s3_class(ms, "SOMAMeasurement")
  expect_true(ms$exists())

  expect_identical(ms$uri, file.path(collection$uri, "rna"))
  expect_identical(ms$names(), c("X", "var"))
  expect_s3_class(ms$var, "SOMADataFrame")
  expect_identical(setdiff(ms$var$attrnames(), "var_id"), names(rna[[]]))
  expect_s3_class(ms$X, "SOMACollection")
  layers <- c(counts = "counts", data = "data", scale.data = "scale_data")
  expect_identical(ms$X$names(), unname(layers))
  for (i in seq_along(layers)) {
    expect_equal(
      ms$X$get(layers[i])$shape(),
      expected = rev(dim(rna)),
      info = layers[i]
    )
  }

  ms$close()
  gc()

  # Test no feature-level meta data
  rna2 <- rna
  for (i in names(rna2[[]])) {
    rna2[[i]] <- NULL
  }
  expect_no_condition(ms2 <- write_soma(rna2, uri = "rna-no-md", soma_parent = collection))
  on.exit(ms2$close(), add = TRUE, after = FALSE)
  expect_s3_class(ms2, "SOMAMeasurement")
  expect_true(ms2$exists())
  expect_identical(ms2$uri, file.path(collection$uri, "rna-no-md"))
  expect_identical(ms2$names(), c("X", "var"))
  expect_s3_class(ms2$var, "SOMADataFrame")
  expect_identical(ms2$var$attrnames(), "var_id")

  ms2$close()
  gc()

  # Test no counts
  rna3 <- SeuratObject::SetAssayData(rna, "counts", new("matrix"))
  expect_no_condition(ms3 <- write_soma(rna3, uri = "rna-no-counts", soma_parent = collection))
  on.exit(ms3$close(), add = TRUE, after = FALSE)
  expect_s3_class(ms3, "SOMAMeasurement")
  expect_true(ms3$exists())
  expect_identical(ms3$uri, file.path(collection$uri, "rna-no-counts"))
  expect_identical(ms3$names(), c("X", "var"))
  expect_s3_class(ms3$X, "SOMACollection")
  # Using a subset of the `layers` map
  lyrs <- layers[c("data", "scale.data")]
  expect_identical(ms3$X$names(), unname(lyrs))
  for (i in seq_along(lyrs)) {
    expect_equal(
      ms3$X$get(lyrs[i])$shape(),
      expected = rev(dim(rna3)),
      info = lyrs[i]
    )
  }

  ms3$close()
  gc()

  # Test no scale.data
  rna4 <- SeuratObject::SetAssayData(rna, "scale.data", new("matrix"))
  expect_no_condition(ms4 <- write_soma(rna4, uri = "rna-no-scale", soma_parent = collection))
  on.exit(ms4$close(), add = TRUE, after = FALSE)
  expect_s3_class(ms4, "SOMAMeasurement")
  expect_true(ms4$exists())
  expect_identical(ms4$uri, file.path(collection$uri, "rna-no-scale"))
  expect_identical(ms4$names(), c("X", "var"))
  expect_s3_class(ms4$X, "SOMACollection")
  # Using a subset of the `layers` map
  lyrs <- layers[c("counts", "data")]
  expect_identical(ms4$X$names(), unname(lyrs))
  for (i in seq_along(lyrs)) {
    expect_equal(
      ms4$X$get(lyrs[i])$shape(),
      expected = rev(dim(rna4)),
      info = lyrs[i]
    )
  }

  ms4$close()
  gc()

  # Test no counts or scale.data
  rna5 <- SeuratObject::SetAssayData(rna3, "scale.data", new("matrix"))
  expect_no_condition(ms5 <- write_soma(rna5, uri = "rna-no-counts-scale", soma_parent = collection))
  on.exit(ms5$close(), add = TRUE, after = FALSE)
  expect_s3_class(ms5, "SOMAMeasurement")
  expect_true(ms5$exists())
  expect_identical(ms5$uri, file.path(collection$uri, "rna-no-counts-scale"))
  expect_identical(ms5$names(), c("X", "var"))
  expect_s3_class(ms5$X, "SOMACollection")
  # Using a subset of the `layers` map
  lyrs <- layers[c("counts", "data")]
  expect_identical(ms5$X$names(), "data")
  expect_equal(ms5$X$get("data")$shape(), rev(dim(rna5)))

  ms5$close()
  gc()

  # Verify data slot isn't ingested when it's identical to counts
  rna6 <- SeuratObject::CreateAssayObject(
    counts = SeuratObject::GetAssayData(rna, "counts")
  )
  expect_identical(
    SeuratObject::GetAssayData(rna6, "counts"),
    SeuratObject::GetAssayData(rna6, "data")
  )
  expect_no_condition(ms6 <- write_soma(
    rna6,
    uri = "rna-identical-counts-data",
    soma_parent = collection
  ))
  on.exit(ms6$close(), add = TRUE, after = FALSE)
  expect_equal(ms6$X$names(), "counts")

  ms6$close()
  gc()

  # Test assertions
  expect_error(write_soma(rna, uri = TRUE, soma_parent = collection))
  expect_error(write_soma(rna, uri = c("dir", "rna"), soma_parent = collection))
  expect_error(write_soma(
    rna,
    soma_parent = SOMADataFrameCreate(uri = file.path(uri, "data-frame"))
  ))

  gc()
})

test_that("Write v5 in-memory Assay mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", minimum_version = "5.0.2")

  uri <- tempfile(pattern = "write-v5-assay")
  collection <- SOMACollectionCreate(uri)
  on.exit(collection$close(), add = TRUE, after = FALSE)

  rna <- get_data("pbmc_small", package = "SeuratObject")[["RNA"]]
  rna <- as(rna, "Assay5")
  expect_no_condition(ms <- write_soma(rna, soma_parent = collection))
  expect_s3_class(ms, 'SOMAMeasurement')
  expect_true(ms$exists())
  on.exit(ms$close(), add = TRUE, after = FALSE)

  expect_identical(ms$uri, file.path(collection$uri, "rna"))
  assay_hint <- .assay_version_hint("v5")
  expect_equivalent(
    ms$get_metadata(names(assay_hint)),
    assay_hint[[1L]]
  )
  expect_identical(ms$names(), c("X", "var"))
  expect_s3_class(ms$var, "SOMADataFrame")
  expect_identical(setdiff(ms$var$attrnames(), "var_id"), names(rna[[]]))
  expect_s3_class(ms$X, "SOMACollection")
  expect_identical(ms$X$names(), SeuratObject::Layers(rna))
  features_map <- methods::slot(rna, name = "features")
  cells_map <- methods::slot(rna, name = "cells")
  ragged_hint <- .ragged_array_hint()
  type_hint <- names(.type_hint(NULL))
  for (layer in SeuratObject::Layers(rna)) {
    idx <- which(cells_map[, layer])
    jdx <- which(features_map[, layer])
    expect_equal(ms$X$get(layer)$shape(), c(max(idx), max(jdx)), info = layer)
    switch(
      EXPR = layer,
      scale.data = expect_equivalent(
        ms$X$get(layer)$get_metadata(names(ragged_hint)),
        ragged_hint[[1L]],
        info = layer
      ),
      expect_null(ms$X$get(layer)$get_metadata(names(ragged_hint)), info = layer)
    )
    expect_type(th <- ms$X$get(layer)$get_metadata(type_hint), 'character')
    expect_length(th, 1L)
    switch(
      EXPR = layer,
      scale.data = expect_identical(th, 'matrix', info = layer),
      expect_true(grepl('^Matrix', x = th), info = layer)
    )
  }

  # Test ragged arrays
  mat <- SeuratObject::LayerData(rna, "counts")
  cells2 <- paste0(colnames(rna), ".2")
  features2 <- paste0(rownames(rna), ".2")
  layers <- list(
    mat = mat,
    cells2 = `colnames<-`(mat, cells2),
    features2 = `rownames<-`(mat, features2),
    c2f2 = `dimnames<-`(mat, list(features2, cells2))
  )
  expect_s4_class(rna2 <- SeuratObject::.CreateStdAssay(layers), "Assay5")
  expect_identical(dim(rna2), dim(rna) * 2)

  expect_no_condition(ms2 <- write_soma(rna2, uri = "ragged-arrays", soma_parent = collection))
  expect_s3_class(ms2, "SOMAMeasurement")
  expect_true(ms2$exists())
  on.exit(ms2$close(), add = TRUE, after = FALSE)

  expect_identical(ms2$uri, file.path(collection$uri, "ragged-arrays"))
  expect_identical(ms2$X$names(), SeuratObject::Layers(rna2))
  features_map <- methods::slot(rna2, name = "features")
  cells_map <- methods::slot(rna2, name = "cells")
  for (layer in SeuratObject::Layers(rna2)) {
    idx <- which(cells_map[, layer])
    jdx <- which(features_map[, layer])
    expect_equal(ms2$X$get(layer)$shape(), c(max(idx), max(jdx)), info = layer)
    expect_equivalent(
      ms2$X$get(layer)$get_metadata(names(ragged_hint)),
      ragged_hint[[1L]],
      info = layer
    )
    expect_true(
      grepl('^Matrix', x = ms2$X$get(layer)$get_metadata(type_hint)),
      info = layer
    )
  }
})

test_that("Write DimReduc mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", .MINIMUM_SEURAT_VERSION("c"))

  uri <- tempfile(pattern = "write-reduction")
  collection <- SOMACollectionCreate(uri)
  on.exit(collection$close(), add = TRUE, after = FALSE)
  pbmc_small <- get_data("pbmc_small", package = "SeuratObject")
  pbmc_small_rna <- pbmc_small[["RNA"]]
  pbmc_small_pca <- pbmc_small[["pca"]]
  pbmc_small_tsne <- pbmc_small[["tsne"]]

  # Test writing PCA
  ms_pca <- write_soma(pbmc_small_rna, uri = "rna-pca", soma_parent = collection)
  on.exit(ms_pca$close(), add = TRUE, after = FALSE)
  fidx <- match(rownames(SeuratObject::Loadings(pbmc_small_pca)), rownames(pbmc_small_rna))
  expect_no_condition(write_soma(
    pbmc_small_pca,
    soma_parent = ms_pca,
    fidx = fidx,
    nfeatures = nrow(pbmc_small_rna)
  ))
  expect_identical(sort(ms_pca$names()), sort(c("X", "var", "obsm", "varm")))
  expect_identical(ms_pca$obsm$names(), "X_pca")
  expect_s3_class(spca <- ms_pca$obsm$get("X_pca"), "SOMASparseNDArray")
  expect_equal(spca$shape(), dim(pbmc_small_pca))
  expect_identical(ms_pca$varm$names(), "PCs")
  expect_s3_class(sldgs <- ms_pca$varm$get("PCs"), "SOMASparseNDArray")
  expect_equal(sldgs$shape(), c(nrow(pbmc_small_rna), ncol(pbmc_small_pca)))

  # Test writing tSNE
  ms_tsne <- write_soma(pbmc_small_rna, uri = "rna-tsne", soma_parent = collection)
  on.exit(ms_tsne$close(), add = TRUE, after = FALSE)
  expect_no_condition(write_soma(pbmc_small_tsne, soma_parent = ms_tsne))
  expect_true(all(ms_tsne$names() %in% c("X", "var", "obsm", "varm")))
  expect_identical(ms_tsne$obsm$names(), "X_tsne")
  expect_s3_class(stsne <- ms_tsne$obsm$get("X_tsne"), "SOMASparseNDArray")
  expect_equal(stsne$shape(), dim(pbmc_small_tsne))
  # Test writing both PCA and tSNE
  ms <- write_soma(pbmc_small_rna, soma_parent = collection)
  expect_no_condition(ms_pca2 <- write_soma(
    pbmc_small_pca,
    soma_parent = ms,
    fidx = fidx,
    nfeatures = nrow(pbmc_small_rna)
  ))
  on.exit(ms_pca2$close(), add = TRUE, after = FALSE)
  expect_no_condition(write_soma(pbmc_small_tsne, soma_parent = ms))
  ms$reopen(ms$mode())
  expect_identical(sort(ms$names()), sort(c("X", "var", "obsm", "varm")))
  expect_identical(sort(ms$obsm$names()), sort(paste0("X_", c("pca", "tsne"))))
  expect_identical(ms$varm$names(), "PCs")

  # Test assertions
  expect_error(write_soma(pbmc_small_pca, uri = "X_pca", soma_parent = ms_tsne))
  expect_error(write_soma(pbmc_small_pca, soma_parent = collection))
  expect_true(ms_tsne$is_open())
  expect_warning(ms_pca3 <- write_soma(pbmc_small_pca, soma_parent = ms_tsne))
  expect_error(ms_tsne$varm)

  gc()
})

test_that("Write Graph mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", .MINIMUM_SEURAT_VERSION("c"))

  uri <- tempfile(pattern = "write-graph")
  collection <- SOMACollectionCreate(uri)
  on.exit(collection$close(), add = TRUE, after = FALSE)

  pbmc_small <- get_data("pbmc_small", package = "SeuratObject")
  pbmc_small_rna <- pbmc_small[["RNA"]]
  graph <- pbmc_small[["RNA_snn"]]

  ms <- write_soma(pbmc_small_rna, soma_parent = collection)
  on.exit(ms$close(), add = TRUE, after = FALSE)
  expect_no_condition(write_soma(graph, uri = "rna-snn", soma_parent = ms))
  expect_identical(sort(ms$names()), sort(c("X", "var", "obsp")))
  expect_identical(ms$obsp$names(), "rna-snn")
  expect_s3_class(sgrph <- ms$obsp$get("rna-snn"), "SOMASparseNDArray")
  expect_equal(sgrph$shape(), dim(graph))

  # Test assertions
  expect_error(write_soma(graph, collection = soma_parent))

  gc()
})

test_that("Write SeuratCommand mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", .MINIMUM_SEURAT_VERSION("c"))
  skip_if_not_installed("jsonlite")

  uri <- tempfile(pattern = "write-command-log")
  uns <- SOMACollectionCreate(uri)
  on.exit(uns$close(), add = TRUE, after = FALSE)

  pbmc_small <- get_data("pbmc_small", package = "SeuratObject")
  for (cmd in SeuratObject::Command(pbmc_small)) {
    cmdlog <- pbmc_small[[cmd]]
    cmdlist <- as.list(cmdlog)
    # Test dumping the command log to SOMA
    expect_no_condition(write_soma(cmdlog, uri = cmd, soma_parent = uns), )
    expect_s3_class(cmdgrp <- uns$get("seurat_commands"), "SOMACollection")

    expect_s3_class(cmddf <- cmdgrp$get(cmd), "SOMADataFrame")
    expect_invisible(cmddf$reopen("READ"))

    # Test qualities of the SOMADataFrame
    expect_identical(cmddf$attrnames(), "values")
    expect_identical(sort(cmddf$colnames()), sort(c("soma_joinid", "values")))
    expect_identical(basename(cmddf$uri), cmd)
    expect_equal(cmddf$ndim(), 1L)

    # Test reading the SOMADataFrame
    expect_s3_class(tbl <- cmddf$read()$concat(), "Table")
    expect_equal(dim(tbl), c(1L, 2L))
    expect_identical(colnames(tbl), cmddf$colnames())
    expect_s3_class(df <- as.data.frame(tbl), "data.frame")
    expect_type(df$values, "character")

    # Test decoding the JSON-encoded command log
    expect_type(vals <- jsonlite::fromJSON(df$values), "list")
    # Test slots of the command log
    for (slot in setdiff(methods::slotNames(cmdlog), "params")) {
      cmdslot <- methods::slot(cmdlog, slot)
      cmdslot <- if (is.null(cmdslot)) {
        cmdslot
      } else if (inherits(cmdslot, "POSIXt")) {
        cmdslot <- as.character(jsonlite::toJSON(
          sapply(
            unclass(as.POSIXlt(cmdslot)),
            .encode_as_char,
            simplify = FALSE,
            USE.NAMES = TRUE
          ),
          auto_unbox = TRUE
        ))
      } else if (is.character(cmdslot)) {
        paste(trimws(cmdslot), collapse = " ")
      } else {
        as.character(cmdslot)
      }
      expect_identical(vals[[slot]], cmdslot)
    }
    # Test encoded parameters
    expect_length(params <- vals[names(cmdlist)], length(cmdlist))
    expect_identical(sort(names(params)), sort(names(cmdlist)))
    for (param in names(params)) {
      if (is.character(cmdlist[[param]])) {
        expect_identical(params[[param]], cmdlist[[param]])
      } else if (is.double(cmdlist[[param]])) {
        # Doubles are encoded as hexadecimal
        expect_identical(params[[param]], sprintf("%a", cmdlist[[param]]))
      } else {
        expect_equivalent(params[[param]], cmdlist[[param]])
      }
    }

    cmddf$close()
    cmdgrp$close()
    gc()
  }

  uns$close()
  gc()
})

test_that("Write Seurat mechanics", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", .MINIMUM_SEURAT_VERSION("c"))

  pbmc_small <- get_data("pbmc_small", package = "SeuratObject")
  uri <- tempfile(pattern = SeuratObject::Project(pbmc_small))

  expect_no_condition(uri <- write_soma(pbmc_small, uri))
  expect_type(uri, "character")
  expect_true(grepl(
    paste0("^", SeuratObject::Project(pbmc_small)),
    basename(uri)
  ))
  expect_no_condition(experiment <- SOMAExperimentOpen(uri))
  on.exit(experiment$close(), add = TRUE, after = FALSE)

  expect_s3_class(experiment, "SOMAExperiment")
  expect_equal(experiment$mode(), "READ")
  expect_true(grepl(
    paste0("^", SeuratObject::Project(pbmc_small)),
    basename(experiment$uri)
  ))

  expect_no_error(experiment$ms)
  expect_identical(experiment$ms$names(), "RNA")
  expect_s3_class(ms <- experiment$ms$get("RNA"), "SOMAMeasurement")
  on.exit(ms$close(), add = TRUE, after = FALSE)

  expect_identical(sort(ms$X$names()), sort(c("counts", "data", "scale_data")))
  expect_identical(sort(ms$obsm$names()), sort(c("X_pca", "X_tsne")))
  expect_identical(ms$varm$names(), "PCs")
  expect_identical(ms$obsp$names(), "RNA_snn")
  expect_error(ms$varp)
  expect_identical(
    setdiff(experiment$obs$attrnames(), "obs_id"),
    names(pbmc_small[[]])
  )

  ms$close()
  experiment$close()

  # Test assertions
  expect_error(write_soma(pbmc_small, TRUE))
  expect_error(write_soma(pbmc_small, 1))
  expect_error(write_soma(pbmc_small, ""))

  gc()
})

test_that("Write Seurat with v3 and v5 assays", {
  skip_if(!extended_tests())
  skip_if_not_installed("SeuratObject", minimum_version = "5.0.2")

  pbmc_small <- get_data("pbmc_small", package = "SeuratObject")
  suppressWarnings(pbmc_small[["RNA5"]] <- methods::as(pbmc_small[["RNA"]], "Assay5"))
  extra <- c(
    SeuratObject::Graphs(pbmc_small),
    SeuratObject::Reductions(pbmc_small),
    SeuratObject::Command(pbmc_small)
  )
  for (i in extra) {
    pbmc_small[[i]] <- NULL
  }

  assay_hint <- .assay_version_hint("v5")
  uri <- tempfile(pattern = SeuratObject::Project(pbmc_small))

  expect_no_condition(uri <- write_soma(pbmc_small, uri))
  expect_type(uri, "character")
  expect_no_condition(experiment <- SOMAExperimentOpen(uri))
  on.exit(experiment$close(), add = TRUE, after = FALSE)

  expect_s3_class(experiment, "SOMAExperiment")
  expect_no_error(experiment$ms)
  expect_identical(sort(experiment$ms$names()), sort(c("RNA", "RNA5")))
  for (assay in experiment$ms$names()) {
    expect_s3_class(experiment$ms$get(assay), "SOMAMeasurement")
    if (inherits(pbmc_small[[assay]], "Assay5")) {
      expect_equivalent(
        experiment$ms$get(assay)$get_metadata(names(assay_hint)),
        assay_hint[[1L]],
        info = assay
      )
    }
  }

  obs_hints <- vapply(
    X = SeuratObject::.FilterObjects(pbmc_small, "Assay5"),
    FUN = .assay_obs_hint,
    FUN.VALUE = character(1L),
    USE.NAMES = FALSE
  )
  expect_true(all(obs_hints %in% experiment$obs$colnames()))
})
