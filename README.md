# Forecasting Floods in Cornwall 

- Motivation : as an aspiring data scientist, I wanted to challenge my end-to-end data science skillset with an ambitious project, whose subject matter is new to me.

- This project I started last year, didn't have the chace to complete. Revisiting

- I found a quality data source, tapping into Gaugemap's API to pull river gauge data

- I want to build an LSTM: either modelling this as a classification problem (flood/ no flood @ cornwall) or modelling water levels as a continuous vairable (Regression)


- End goal: deploy a trained LSTM model via a web app (Streamlit / Mesop?), try Dockerisation. Try Cloud modelling. Will be an online learning model. 




-----
## URLs for APIs:
- https://www.gaugemap.co.uk/
-  weather API hasn't been finalised yet: testing Meteostat (although there is some missing data). I can pull hourly data and daily data for regular weather stats.
- meteostat.com 


## Stage 1: Data Collection 

I have successfully pulled in 20 years of river gauge and weather data for Cornwall stations (missing some river stations due to unavailability of data)

Progress : river data is DONE, weather data is DONE (subject to revision)

- Pulled gauge data from 2000 1st Jan until 11 July 2024



## Stage 2 : ETL 

Coding the functions for ETL, adhering to OOP best practices. Once the data is clean and all the functions for cleaning, extracting, transforming are created I can proceed to next stage. DONE

-Note : Use Google Cloud/ AWS to store data? Opens up other cloud functionality. 

- Need a function to merge river + weather data temporally (it is already matched location-wise)


## Stage 3 : EDA/ Feature Engineering

- Quality control : missing values,...,
- Investigate dispersion of river gauge measurements, weather columns 
- This is where I'm going to narrow down the models I'm going to train 


Potential tech stack 

<!-- Technologies and Frameworks
Frameworks: TensorFlow, PyTorch, and Scikit-learn (for initial model training and simpler online learning tasks).
Streaming Platforms: Apache Kafka, Apache Flink, and AWS Kinesis for managing data streams.
Cloud Services: AWS, Azure, and Google Cloud offer capabilities for deploying and managing real-time machine learning applications. -->


Data alignment: temporal 
