# Data dashboard for Metro21 Fire Risk Analysis Project
# Created for: Pittsburgh Bureau of Fire
# Authors: Qianyi Hu, Michael Madaio, Geoffrey Arnold
# Latest update: February 20, 2018

# Install Packages
# install.packages("shiny")
# install.packages("dplyr")
# install.packages("ggplot2")

# Load Packages
library(shiny)
library(markdown)
library(ggplot2)
library(shiny)
library(dplyr)
library(plotly)
library(jsonlite)
library(R4CouchDB)
library(httr)

# User settings
userName <- as.character(Sys.getenv("SHINYPROXY_USERNAME"))
bookmark_id <- paste0(userName, "-pbf-firerisk-dashboard")

grabInput <- function(default = "", inputs = inputs, id) {
  if (is.null(inputs)) {
    selected <- default
  } else if (is.na(inputs[id])) {
    selected <- default
  } else {
    selected <- unlist(inputs[id])
  }
  return(selected)
}

selectGet <- function(id, conn) {
  rurl <- paste0("http://", keys$mcndb_url, ":5984/proxy-user-settings/", id)
  rg <- GET(rurl, authenticate(keys$mcndb_un, keys$mcndb_pw))
  if (rg$status_code != 200) {
    r <- NULL
  } else {
    conn$id <- id
    r <- cdbGetDoc(conn)$res
  }
  return(r)
}

addUpdateDoc <- function (id, data, conn) {
  conn$dataList <- data
  conn$id <- id
  rurl <- paste0("http://", keys$mcndb_url, ":5984/proxy-user-settings/", id)
  rg <- GET(rurl, authenticate(keys$mcndb_un, keys$mcndb_pw))
  if (is.null(rg$header$etag)) {
    cdbAddDoc(conn)
  } else {
    cdbUpdateDoc(conn)
  }
}

#Key
keys <- jsonlite::fromJSON(".key.json")

# Grab Inputs
conn <- cdbIni(serverName = keys$mcndb_url, uname = keys$mcndb_un, pwd = keys$mcndb_pw, DBName = "proxy-user-settings")

inputs <- selectGet(bookmark_id, conn)

# Read data  
loadModel <- read.csv("http://rstudio.city.pittsburgh.pa.us/fire_risk_nonres.csv")   