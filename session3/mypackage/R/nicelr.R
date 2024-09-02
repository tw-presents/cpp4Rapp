

nicelr <- function(y, x) {
  
  stopifnot("argument y must be a matrix" = is.matrix(y))
  
  qqq = .Call(`_mypackage_nicelr`, y, x)
  
  return(qqq)
}

