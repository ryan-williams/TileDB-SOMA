.pkgenv <- new.env(parent = emptyenv())

## .onLoad is called whether code from the package is used and the packages is 'loaded'. An
## example is calling `tiledbsoma::show_package_versions()`. So this is most elementary check,
## .onAttach is also called when the package is 'attached' via 'library(tiledbsoma)'
## During package build and byte-code compilation and load check, both are called.
.onLoad <- function(libname, pkgname) {
  ## create a slot for somactx in per-package enviroment, do no fill it yet to allow 'lazy load'
  .pkgenv[["somactx"]] <- NULL

  rpkg_lib <- tiledb::tiledb_version(compact = FALSE)
  # Check major and minor but not micro: sc-50464
  rpkg_lib_version <- paste(rpkg_lib[["major"]], rpkg_lib[["minor"]], sep = ".")
  soma_lib_version <- libtiledbsoma_version(compact = TRUE, major_minor_only = TRUE)
  if (rpkg_lib_version != soma_lib_version) {
    msg <- sprintf(
      "TileDB Core version %s used by TileDB-R package, but TileDB-SOMA uses %s",
      sQuote(rpkg_lib_version), sQuote(soma_lib_version)
    )
    packageStartupMessage(msg)
  }
}

## An .onAttach() function is not allowed to use cat() etc but _must_ communicate via
## packageStartupMessage() as this function can be 'muzzled' as desired. See Writing R Extensions.
.onAttach <- function(libname, pkgname) {
  if (interactive()) {
    packageStartupMessage(
      "TileDB-SOMA R package ", packageVersion(pkgname),
      " with TileDB Embedded ", format(tiledb::tiledb_version(TRUE)),
      " on ", utils::osVersion,
      ".\nSee https://github.com/single-cell-data for more information ",
      "about the SOMA project."
    )
  }
}

#' Create and cache a SOMA Context Object
#'
#' @param config A named character vector with \sQuote{key} and \sQuote{value} pairs defining the
#' configuration setting
#' @return An external pointer object containing a shared pointer instance of \code{SOMAContext}
#' @examples
#' cfgvec <- as.vector(tiledb::tiledb_config()) # TileDB Config in vector form
#' sctx <- soma_context(cfgvec)
#' @export
soma_context <- function(config) {
  ## if a new config is given always create a new object
  if (!missing(config)) {
    somactx <- createSOMAContext(config)
    .pkgenv[["somactx"]] <- somactx
  }

  ## access config
  somactx <- .pkgenv[["somactx"]]

  ## if no values was cached, create a new one with either empty or given config
  if (is.null(somactx)) {
    if (missing(config)) {
      somactx <- createSOMAContext()
    } else {
      somactx <- createSOMAContext(config)
    }
    .pkgenv[["somactx"]] <- somactx
  }

  return(somactx)
}
