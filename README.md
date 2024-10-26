# Flood Prediction System for Cornwall, UK

Welcome to the **Flood Prediction System for Cornwall, UK** – a real-time flood prediction project aimed at providing actionable flood risk insights for a region prone to flooding. This system is designed to ingest river gauge and weather data, process and clean it, model flood risks, and ultimately deploy an interactive flood prediction service.

---

## Project Overview 🌊

Cornwall, UK, is a region known for its flood risk, which inspired this project to predict river flood events in real-time. By leveraging various open-source datasets, APIs, and machine learning, this system seeks to deliver accurate and timely flood predictions for communities across the region.

## Objectives 🏆

1. **Real-time Flood Prediction**: Develop a model that predicts river flooding with reliable accuracy for 30 river stations across Cornwall.
2. **Scalable Deployment**: Deploy the model on AWS, enabling users to interact with it in real-time and visualize live predictions.
3. **Comprehensive Workflow**: Build an end-to-end solution from data ingestion to model deployment, demonstrating a full data science pipeline.
4. **Community Engagement**: Share insights, challenges, and progress through weekly video updates on LinkedIn, fostering feedback and connecting with industry professionals.

## Features Implemented 🔧

### Data Collection

- **River Gauge Levels**: 
  - Pulled 10 years of river gauge levels across 30 stations in Cornwall from Shoothill Gauge Map API.
  - Data transformed from JSON to clean, structured CSV files with consistent time-stamps.
- **Weather Data**:
  - Aggregated historical weather observations from NOAA for Cornwall’s available station.
  - Explored several open-source APIs to find reliable, consistent weather data sources.

### Data Cleaning & Transformation

- Extensive data cleaning, including handling missing data, time-stamping inconsistencies, and data normalization.
- All river gauge and weather data are consolidated into CSV files for ease of analysis and reproducibility.

### Exploratory Data Analysis (EDA)

- **Data Visualizations**:
  - Interactive visualizations to understand seasonal and daily river level patterns.
  - Analysis of weather and river gauge correlations to inform feature engineering.
- **Time-Series EDA**:
  - Investigating river level trends to identify potential flood indicators.
  - Analyzing missing data patterns and examining impact on model accuracy.

## Roadmap 🚀

1. **Modeling**: Building baseline regression and classification models, with plans to incorporate time-series modeling for river level prediction.
2. **Threshold-Based Classification**: Convert continuous river level predictions into flood/no-flood classifications.
3. **Deployment**: Deploy a working model on AWS using Docker and serverless architecture to provide a real-time prediction service.
4. **Real-Time Data Pipeline**: Establish an automated pipeline to continuously ingest new data for model re-training and prediction.

## Weekly LinkedIn Updates 📅

Follow my journey on LinkedIn, where I’ll be posting regular updates and videos detailing each phase of the project, from data collection to deployment. These posts are intended to showcase technical skills, foster community feedback, and build engagement with the data science network.

## Repository Structure 📁

Here’s an overview of the project’s file structure:
```
.
├── assets
├── data
│   ├── river_data
│   └── weather_data
├── notebooks
├── src
└── tests

```
- **`data/`**: Contains the cleaned CSV files of river gauge and weather data.
- **`src/`**: Python scripts for data ingestion, cleaning, and initial EDA.
- **`notebooks/`**: Jupyter notebooks documenting EDA, modeling, and feature engineering.
- **`README.md`**: Project overview, goals, and roadmap (you are here).
- **`assets/`**: Screenshots and visualizations for future documentation.

---

## Next Steps 🎯

As the project evolves, I’ll be refining the model, optimizing feature engineering, and preparing for deployment. The goal is to create a robust, real-time flood prediction tool that effectively demonstrates data engineering, machine learning, and model deployment in a professional context.

Stay tuned for more updates on LinkedIn and feel free to connect for feedback, discussions, or collaboration. Let’s make real-time flood prediction a reality!

---

**Author**: Misha  
**Project Status**: 🚧 In Progress  
**Contact**: Connect with me on [LinkedIn](https://www.linkedin.com/in/misha-freidin/)  

---


