# SABRE cross-check for the report's regionalizations.
#
# Purpose: read the K = 3 class-label rasters that Python exported to
# outputs/premodel/sabre_input/{var}_{res}.tif, turn each into an sf polygon
# regionalization, and let the CRAN package 'sabre' compute its own V-measure,
# homogeneity and completeness for every resolution pair. The numbers are then
# compared against the direct Python computation in
# outputs/premodel/sabre_python_vmeasure.json.
#
# Run with the dedicated conda environment:
#   conda env r-sabre (r-base 4.5.3, r-sf, r-terra, sabre 0.4.3)
#
# The result CSV is rewritten after every pair so a partial run still yields
# usable evidence.

suppressMessages({
  library(terra)
  library(sf)
  library(sabre)
})

root <- "E:/Projects/20260921-JiaChenchen-Paper/outputs/premodel/sabre_input"
out_csv <- "E:/Projects/20260921-JiaChenchen-Paper/outputs/premodel/sabre_r_vmeasure.csv"
out_log <- "E:/Projects/20260921-JiaChenchen-Paper/outputs/premodel/sabre_r_run.log"

res <- c("2m", "5m", "10m", "20m", "30m")
vars <- c("h_max", "speed")

logline <- function(...) {
  txt <- paste0(..., collapse = " ")
  cat(txt, "\n", file = out_log, append = TRUE)
  flush.console()
}

cat("sabre cross-check start\n", file = out_log)
logline("R", R.version.string, "| sabre", as.character(packageVersion("sabre")),
        "| sf", as.character(packageVersion("sf")),
        "| terra", as.character(packageVersion("terra")))

as_region <- function(path) {
  r <- rast(path)
  s <- st_as_sf(as.polygons(r, dissolve = TRUE, values = TRUE))
  names(s)[1] <- "z"
  s$z <- as.character(s$z)
  s
}

rows <- list()
write_rows <- function() {
  if (length(rows)) {
    write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
  }
}

for (var in vars) {
  polys <- lapply(res, function(x) as_region(file.path(root, paste0(var, "_", x, ".tif"))))
  names(polys) <- res
  logline("[sabre]", var, "regions", paste(sapply(polys, nrow), collapse = "/"))
  for (i in 1:4) {
    for (j in (i + 1):5) {
      a <- res[i]; b <- res[j]
      t0 <- Sys.time()
      v <- vmeasure_calc(x = polys[[a]], y = polys[[b]], x_name = z, y_name = z)
      el <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
      vm <- as.numeric(v$v_measure)
      rows[[length(rows) + 1]] <- data.frame(
        var = var, a = a, b = b, pair = paste0(a, "|", b),
        v = vm, h = as.numeric(v$homogeneity),
        c = as.numeric(v$completeness), elapsed_s = el)
      write_rows()
      logline(sprintf("[sabre] %s %s|%s v=%.9f h=%.9f c=%.9f (%.1fs)",
                      var, a, b, vm, as.numeric(v$homogeneity),
                      as.numeric(v$completeness), el))
    }
  }
  rm(polys); gc()
}
logline("wrote", out_csv)
