# 
# Purpose: To take a look at the soils data over time
# Author: Jake Diamond + Andrew Merdith
# Date: 7 November 2022
# 

# Load libraries
library(tidyverse)
library(tidytable)

# Load all soils---------------------------------------------------------------
# Data directory
data_dir <- file.path("data", "00_soils")

# files
files <- fs::dir_ls(data_dir, regexp = "\\.csv$")

# Load soils data
df_soils <- files %>% 
  map_dfr(read_csv, .id = "filename") %>%
  janitor::clean_names()

# Save this
write_csv(df_soils, file.path("data", "soils", "all_soil_data.csv"))

# Read in the pySCION timeseries of landmasses
df_land <- R.matlab::readMat(file.path("data", "01_spatial", "INTERPSTACK2.mat"))

# Quick plots -------------------------------------------------------------
# Look at distribution of soil types
ggplot(data = df_soils,
      aes(x = lithology)) +
  geom_bar(aes(fill = period)) + 
  scale_y_log10() +
  theme_classic()

# Do it in space
df_soils_sf <- st_as_sf(df_soils_sf)


