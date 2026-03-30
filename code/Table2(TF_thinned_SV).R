suppressPackageStartupMessages({
  library(data.table)
  library(GpGp)
})

ROOT <- normalizePath(getwd())
if (!dir.exists(file.path(ROOT, "data"))) {
  ROOT <- normalizePath(file.path(ROOT, ".."))
}

DATA_DIR <- file.path(ROOT, "data")
PROCESSED_DIR <- file.path(DATA_DIR, "processed_data")
RESULTS_DIR <- file.path(ROOT, "results", "intermediate")
CODE_DIR <- file.path(ROOT, "code")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

source(file.path(CODE_DIR, "thinnedsv_source.R"))

DONOR_FILE <- file.path(PROCESSED_DIR, "matching_weighted_ks.csv")
OUTPUT_DETAIL <- file.path(RESULTS_DIR, "Table2_TF_thinnedSV_detail.csv")
OUTPUT_SUMMARY <- file.path(RESULTS_DIR, "Table2_TF_thinnedSV_summary.csv")

TARGET <- "power"
BASE_FEATURES <- c("wind_speed", "temperature", "turbulence_intensity", "std_wind_direction")
ANGLE_FEATURE <- "wind_direction"
K <- 7L
SEED <- 2026L
MAX_THINNING_NUMBER <- 20L
YEARS_TEST <- c(2017L, 2018L)

mae_vec <- function(y, yhat) mean(abs(y - yhat))
rmse_vec <- function(y, yhat) sqrt(mean((y - yhat)^2))

feature_names <- function() {
  c(BASE_FEATURES, "wind_direction_sin", "wind_direction_cos")
}

load_turbine_year <- function(turbine_id, year) {
  path <- file.path(DATA_DIR, sprintf("Turbine%d_%d.csv", as.integer(turbine_id), as.integer(year)))
  if (!file.exists(path)) stop("Missing file: ", path)
  
  dt <- fread(path, showProgress = FALSE)
  need <- c(BASE_FEATURES, ANGLE_FEATURE, TARGET)
  miss <- setdiff(need, names(dt))
  if (length(miss)) stop("Missing columns in ", path, ": ", paste(miss, collapse = ", "))
  
  dt <- dt[, ..need]
  dt[, (need) := lapply(.SD, as.numeric), .SDcols = need]
  dt <- na.omit(dt)
  if (!nrow(dt)) stop("No usable rows in ", path)
  
  rad <- dt[[ANGLE_FEATURE]] * pi / 180
  dt[, wind_direction_sin := sin(rad)]
  dt[, wind_direction_cos := cos(rad)]
  dt[, (ANGLE_FEATURE) := NULL]
  dt
}

read_donor_table <- function(path) {
  dt <- fread(path, showProgress = FALSE)
  if (names(dt)[1] != "target") setnames(dt, 1L, "target")
  
  donor_cols <- grep("^donor", names(dt), value = TRUE)
  if (!length(donor_cols)) stop("No donor columns found in ", path)
  
  ord <- order(as.integer(gsub("^donor", "", donor_cols)))
  donor_cols <- donor_cols[ord]
  
  dt[, target := as.integer(target)]
  dt[, (donor_cols) := lapply(.SD, as.integer), .SDcols = donor_cols]
  
  list(dt = dt, donor_cols = donor_cols)
}

compute_thinning_number <- function(trainX, max_thinning_number) {
  n <- nrow(trainX)
  if (n < 5L) return(1L)
  
  thinning_vec <- rep(max_thinning_number, ncol(trainX))
  for (j in seq_len(ncol(trainX))) {
    pacf_vals <- tryCatch(
      stats::pacf(trainX[, j], plot = FALSE, lag.max = max_thinning_number)$acf[, 1, 1],
      error = function(e) rep(0, max_thinning_number)
    )
    thresh <- 2 / sqrt(n)
    idx <- which(c(1, abs(pacf_vals)) <= thresh)
    if (length(idx)) thinning_vec[j] <- min(idx)
  }
  max(1L, max(thinning_vec))
}

create_thinned_bins <- function(dataX, dataY, thinning_number) {
  n <- nrow(dataX)
  bins <- vector("list", thinning_number)
  
  if (thinning_number < 2L) {
    bins[[1]] <- list(X = dataX, y = dataY)
    return(bins)
  }
  
  for (i in seq_len(thinning_number)) {
    n_points <- floor((n - i) / thinning_number)
    last_idx <- i + n_points * thinning_number
    idx <- seq(i, last_idx, length.out = n_points + 1L)
    bins[[i]] <- list(X = dataX[idx, , drop = FALSE], y = dataY[idx])
  }
  
  bins
}

thinned_sv_predict <- function(x_train, y_train, x_test, T_use) {
  bins <- create_thinned_bins(x_train, y_train, T_use)
  
  set.seed(SEED)
  fit_obj <- fit_scaled_thinned(
    y = y_train,
    inputs = x_train,
    thinnedBins = bins,
    T = T_use,
    ms = 30
  )
  
  predictions_scaled_thinned(
    fit_obj,
    locs_pred = x_test,
    m = 200,
    joint = TRUE,
    nsims = 0,
    predvar = FALSE,
    scale = "parms"
  )
}

fit_one_donor <- function(donor_id, target_id, test_year) {
  feats <- feature_names()
  
  train_dt <- load_turbine_year(donor_id, 2017L)
  test_dt <- load_turbine_year(target_id, test_year)
  
  x_train <- as.matrix(train_dt[, ..feats])
  y_train <- train_dt[[TARGET]]
  x_test <- as.matrix(test_dt[, ..feats])
  y_test <- test_dt[[TARGET]]
  
  T_use <- compute_thinning_number(x_train, MAX_THINNING_NUMBER)
  
  t0 <- proc.time()[["elapsed"]]
  pred <- thinned_sv_predict(x_train, y_train, x_test, T_use)
  runtime <- proc.time()[["elapsed"]] - t0
  
  list(
    pred = pred,
    actual = y_test,
    rmse = rmse_vec(y_test, pred),
    mae = mae_vec(y_test, pred),
    runtime = runtime,
    T = T_use
  )
}

run_metric <- function() {
  donor_obj <- read_donor_table(DONOR_FILE)
  donor_dt <- donor_obj$dt
  donor_cols <- donor_obj$donor_cols
  
  targets <- sort(unique(donor_dt$target))
  rows <- vector("list", length(targets) * length(YEARS_TEST))
  idx <- 1L
  
  for (target_id in targets) {
    donor_row <- donor_dt[target == target_id]
    donors <- as.integer(na.omit(unlist(donor_row[, ..donor_cols])))
    donors <- unique(donors[donors != target_id])
    donors <- donors[seq_len(min(K, length(donors)))]
    if (!length(donors)) next
    
    for (test_year in YEARS_TEST) {
      pred_list <- list()
      runtime_vec <- numeric(0)
      rmse_vec_single <- numeric(0)
      mae_vec_single <- numeric(0)
      T_vec <- integer(0)
      actual <- NULL
      donors_used <- integer(0)
      
      for (donor_id in donors) {
        res <- tryCatch(
          fit_one_donor(donor_id, target_id, test_year),
          error = function(e) NULL
        )
        if (is.null(res)) next
        
        pred_list[[length(pred_list) + 1L]] <- res$pred
        actual <- res$actual
        runtime_vec <- c(runtime_vec, res$runtime)
        rmse_vec_single <- c(rmse_vec_single, res$rmse)
        mae_vec_single <- c(mae_vec_single, res$mae)
        T_vec <- c(T_vec, res$T)
        donors_used <- c(donors_used, donor_id)
      }
      
      if (!length(pred_list)) next
      
      ensemble_pred <- Reduce(`+`, pred_list) / length(pred_list)
      
      rows[[idx]] <- data.table(
        method = "TF_thinnedSV",
        target = target_id,
        year = test_year,
        donors_used = paste(donors_used, collapse = ","),
        n_models = length(pred_list),
        rmse = rmse_vec(actual, ensemble_pred),
        mae = mae_vec(actual, ensemble_pred),
        runtime_sec = sum(runtime_vec),
        mean_single_rmse = mean(rmse_vec_single),
        mean_single_mae = mean(mae_vec_single),
        mean_T = mean(T_vec)
      )
      idx <- idx + 1L
    }
  }
  
  detail_dt <- rbindlist(rows[seq_len(idx - 1L)], fill = TRUE)
  fwrite(detail_dt, OUTPUT_DETAIL)
  
  summary_dt <- detail_dt[, .(
    avg_rmse = mean(rmse),
    avg_mae = mean(mae),
    total_runtime_sec = sum(runtime_sec)
  ), by = .(method, year)]
  
  fwrite(summary_dt, OUTPUT_SUMMARY)
}

run_metric()