suppressPackageStartupMessages({
  library(data.table)
  library(e1071)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

DONOR_FILE <- file.path(PROCESSED_DIR, "matching_geographic_distance.csv")
OUTPUT_DETAIL <- file.path(RESULTS_DIR, "Table2_G_SVR_detail.csv")
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, "Table2_G_SVR_summary.csv")

TARGET <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"

K <- 7L
TRAIN_YEAR <- 2017L
TEST_YEARS <- c(2017L, 2018L)

SVM_KERNEL <- "radial"
SVM_COST <- 1
SVM_GAMMA <- 1 / 6

rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))

feature_names <- function() {
  c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) return(NULL)
  
  dt <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns in ", path, ": ", paste(miss, collapse = ", "))
  
  dt <- dt[, ..need]
  dt[, (need) := lapply(.SD, as.numeric), .SDcols = need]
  dt <- na.omit(dt)
  if (!nrow(dt)) return(NULL)
  
  rad <- dt[[ANGLE_FEATURE]] * pi / 180
  dt[, wind_direction_sin := sin(rad)]
  dt[, wind_direction_cos := cos(rad)]
  dt[, (ANGLE_FEATURE) := NULL]
  dt
}

read_geo_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  req <- c("target", "donor", "geo_distance")
  miss <- setdiff(req, names(dt))
  if (length(miss)) stop("Missing columns in donor file: ", paste(miss, collapse = ", "))
  
  dt[, target := as.integer(target)]
  dt[, donor := as.integer(donor)]
  dt[, geo_distance := as.numeric(geo_distance)]
  dt <- dt[!is.na(target) & !is.na(donor) & !is.na(geo_distance)]
  setorder(dt, target, geo_distance, donor)
  dt
}

get_top_k_donors <- function(dt, target_id, k) {
  sub <- dt[target == target_id & donor != target_id]
  head(sub$donor, k)
}

fit_one_neighbor_svm <- function(donor_id) {
  feats <- feature_names()
  train_dt <- load_turbine_year(donor_id, TRAIN_YEAR)
  if (is.null(train_dt)) return(NULL)
  
  x_train <- train_dt[, ..feats]
  y_train <- train_dt[[TARGET]]
  
  t0 <- proc.time()[["elapsed"]]
  model <- svm(
    x = x_train,
    y = y_train,
    type = "eps-regression",
    kernel = SVM_KERNEL,
    cost = SVM_COST,
    gamma = SVM_GAMMA,
    scale = TRUE
  )
  fit_time <- proc.time()[["elapsed"]] - t0
  
  list(model = model, fit_time = fit_time)
}

run_target_year <- function(target_id, year, donor_ids) {
  feats <- feature_names()
  test_dt <- load_turbine_year(target_id, year)
  if (is.null(test_dt)) return(NULL)
  
  x_test <- test_dt[, ..feats]
  y_test <- test_dt[[TARGET]]
  
  models <- list()
  fit_times <- numeric(0)
  donors_used <- integer(0)
  
  for (d in donor_ids) {
    res <- tryCatch(fit_one_neighbor_svm(d), error = function(e) NULL)
    if (is.null(res)) next
    models[[length(models) + 1L]] <- res$model
    fit_times <- c(fit_times, res$fit_time)
    donors_used <- c(donors_used, d)
  }
  
  if (!length(models)) return(NULL)
  
  t0 <- proc.time()[["elapsed"]]
  pred_mat <- matrix(NA_real_, nrow = nrow(x_test), ncol = length(models))
  for (j in seq_along(models)) {
    pred_mat[, j] <- predict(models[[j]], newdata = x_test)
  }
  pred <- rowMeans(pred_mat, na.rm = TRUE)
  pred_time <- proc.time()[["elapsed"]] - t0
  
  data.table(
    method = "G_SVR",
    target = target_id,
    year = year,
    donors_used = paste(donors_used, collapse = ","),
    n_models = length(models),
    rmse = rmse_vec(y_test, pred),
    runtime_sec = sum(fit_times) + pred_time,
    fit_time_sec = sum(fit_times),
    pred_time_sec = pred_time
  )
}

main <- function() {
  geo_dt <- read_geo_donor_table(DONOR_FILE)
  targets <- sort(unique(geo_dt$target))
  
  rows <- list()
  idx <- 1L
  
  for (target_id in targets) {
    donor_ids <- get_top_k_donors(geo_dt, target_id, K)
    if (!length(donor_ids)) next
    
    for (year in TEST_YEARS) {
      cat("G_SVR - Turbine", target_id, "Year", year, "\n")
      out <- tryCatch(run_target_year(target_id, year, donor_ids), error = function(e) NULL)
      if (is.null(out)) next
      rows[[idx]] <- out
      idx <- idx + 1L
    }
  }
  
  detail_dt <- rbindlist(rows, fill = TRUE)
  fwrite(detail_dt, OUTPUT_DETAIL)
  
  summary_dt <- detail_dt[, .(
    avg_rmse = mean(rmse, na.rm = TRUE),
    total_runtime_sec = sum(runtime_sec, na.rm = TRUE)
  ), by = .(method, year)]
  
  fwrite(summary_dt, OUTPUT_SUMMARY)
}

main()