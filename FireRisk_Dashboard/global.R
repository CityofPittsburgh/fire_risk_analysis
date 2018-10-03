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

# read data  
loadModel <- read.csv("http://rstudio.city.pittsburgh.pa.us/fire_risk_nonres.csv")   
