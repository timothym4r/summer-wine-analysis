#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

usage <- paste(
  "Usage:",
  "  Rscript modules/rdata_to_csv.R input.RData output.csv [object_name]",
  "",
  "If the file contains the Vivino rating objects, omitting object_name writes",
  "a tidy ratings CSV with user_id, wine_id, rating, and split columns.",
  "",
  "Otherwise, omitting object_name writes the first data frame, matrix, or 2D",
  "array found in the .RData file.",
  sep = "\n"
)

if (length(args) < 2 || length(args) > 3) {
  stop(usage, call. = FALSE)
}

input_path <- args[[1]]
output_path <- args[[2]]
object_name <- if (length(args) == 3) args[[3]] else NULL

if (!file.exists(input_path)) {
  stop("Input file does not exist: ", input_path, call. = FALSE)
}

env <- new.env(parent = emptyenv())
loaded_objects <- load(input_path, envir = env)

has_vivino_rating_data <- all(
  c("set.train", "set.valid", "set.test", "valid.position", "test.position") %in%
    loaded_objects
)

positions_to_ratings <- function(rating_matrix, positions, ratings, split) {
  indices <- arrayInd(positions, .dim = dim(rating_matrix))

  data.frame(
    user_id = rownames(rating_matrix)[indices[, 1]],
    wine_id = colnames(rating_matrix)[indices[, 2]],
    rating = as.numeric(ratings),
    split = split,
    stringsAsFactors = FALSE
  )
}

if (!is.null(object_name)) {
  if (!object_name %in% loaded_objects) {
    stop(
      "Object '", object_name, "' was not found in ",
      input_path, ". Available objects: ",
      paste(loaded_objects, collapse = ", "),
      call. = FALSE
    )
  }

  data <- get(object_name, envir = env)
} else if (has_vivino_rating_data) {
  rating_matrix <- get("set.train", envir = env)

  train_positions <- which(!is.na(rating_matrix))
  valid_positions <- get("valid.position", envir = env)
  test_positions <- get("test.position", envir = env)

  data <- rbind(
    positions_to_ratings(
      rating_matrix,
      train_positions,
      rating_matrix[train_positions],
      "train"
    ),
    positions_to_ratings(
      rating_matrix,
      valid_positions,
      get("set.valid", envir = env),
      "valid"
    ),
    positions_to_ratings(
      rating_matrix,
      test_positions,
      get("set.test", envir = env),
      "test"
    )
  )
} else {
  tabular_names <- loaded_objects[
    vapply(loaded_objects, function(name) {
      value <- get(name, envir = env)
      is.data.frame(value) || is.matrix(value) || length(dim(value)) == 2
    }, logical(1))
  ]

  if (length(tabular_names) == 0) {
    stop(
      "No data frame, matrix, or 2D array objects found in: ",
      input_path,
      call. = FALSE
    )
  }

  if (length(tabular_names) > 1) {
    warning(
      "Multiple tabular objects found; writing the first one: ",
      tabular_names[[1]],
      call. = FALSE
    )
  }

  data <- get(tabular_names[[1]], envir = env)
}

if (!(is.data.frame(data) || is.matrix(data) || length(dim(data)) == 2)) {
  stop("Selected object is not a data frame, matrix, or 2D array.", call. = FALSE)
}

write.csv(as.data.frame(data), output_path, row.names = FALSE)
message("Wrote CSV: ", output_path)
