<!--
Author: NTT Data
Version: 1.0.0
Creation date: 12/04/2026
-->

<a id="readme-top"></a>



<br />
<div align="center">
  <a href="">
    <picture>
      <img src="wiki/img/logo.png" alt="logo" width="200">
    </picture>
  </a>

<h1 align="center">HOU53-bot</h1>

  <p align="center">
    <i>HOU53-bot</i> will help you find a fair price for any house based on your description! The problem is, we just have the data, but <i>HOU53-bot</i> has yet to learn about it.
    <br /><br />
    Keep on reading to learn more about this exciting challenge! 
    <br />
  </p>

![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![GitHub License](https://img.shields.io/github/license/Yagouus/hou53-bot)
![GitHub Release](https://img.shields.io/github/v/release/Yagouus/hou53-bot)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Yagouus/hou53-bot)

</div>


>[!WARNING] 
> **Disclaimer**: This is an educational purpose repo used as a challenge for students in *NTT Data (A Coruña office)*. The original challenge and dataset are extracted from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and an adaptation by [INRIA](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html) will be used. Feel free to use this for educational purposes just by crediting this original source and Kaggle. For any inquire reach out to: yago.fontenlaseco@nttdata.com or angel.fragavarela@nttdata.com.


# 📖 About the challenge

The **objective of this challenge** is to build an end-to-end Machine Learning system for house price prediction that goes beyond a notebook and resembles a real production application. Participants must analyze the dataset, design a preprocessing pipeline, and train and evaluate a predictive model with good generalization performance and interpretability. The trained model must then be packaged and deployed, exposing its functionality through a robust API capable of receiving input data and returning predictions. In addition, participants must incorporate explainability techniques (e.g., feature importance) to justify predictions. Finally, a web-based frontend must be developed where users can provide a natural language description of a house; the system should parse this input into structured features, query the API, and display both the predicted price and an understandable explanation of the result.

<br />

<picture>
      <img src="wiki/img/cover.png" alt="logo" width="800">
</picture>
<br/><br/>

Throughout the challenge, you will apply your knowledge of data science, machine learning, software architecture, and intelligent systems to build a complete solution that predicts house prices. The goal is to help users assess whether a property is fairly priced or estimate the value of their own home when selling.

>[!NOTE] 
> *HOU53-bot* is your personal house valuation assistant. It helps you estimate the price of your dream home and decide whether it’s worth buying. Trained on a rich dataset of real estate properties, *HOU53-bot* learns patterns between house features and their market prices. Through a simple web interface, users can provide a textual description of a property, and the system will generate a price estimation along with insights to support the decision-making process.


The solution to the proposed problem is **open-ended**, and there are multiple ways to approach it. Feel free to explore one or all of the alternatives that you find interesting, as long as they are coherent and you can properly motivate and justify their use.

To participate, you will need to create an account on Kaggle, an online platform where datasets and problems based on those datasets are published in the form of “competitions.” This allows anyone to take part and propose the best possible solution, with many competitions offering monetary prizes for the top solutions.

For this challenge, you will work with a dataset designed for learning purposes. In the future, you will also be able to participate in official challenges hosted on the platform.


## ⚙️ Technical Requirements

To successfully complete this challenge, your solution must include the following components:

**Data Analysis & Modeling**
- Perform an exploratory data analysis (EDA) of the dataset to understand its structure, features, and potential issues.
- Design and implement a data preprocessing pipeline (feature engineering, handling missing values, encoding, etc.).
- Train and evaluate at least one ML model capable of predicting house prices. Justify your modeling choices and evaluation strategy.
- Once you have a trained model, you must evaluate its accuracy and generalization capability. It is important to consider not only the model’s performance but also its interpretability,

**Model Deployment**
- Package your trained model so it can be used in a production environment.
 Ensure the model can be loaded and used for inference independently of notebooks.

**Explainability**
- Incorporate model explainability into your system.
- Provide feature importance or similar interpretability methods to explain predictions.
- Explanations should be understandable to non-technical users.

**API Development**
- Develop a backend API (e.g., REST API) that exposes an endpoint to:
    - Receive input data
    - Return predicted house prices
- The API should be robust, handling invalid or incomplete inputs gracefully.

**Frontend Application**
- Develop a web-based user interface that allows users to:
    - Input house information
    - Receive a predicted price
    - View an explanation of the prediction

**Natural Language Input**
- The system must accept natural language descriptions of a house (e.g., “A 3-bedroom house with a large garden in a suburban area…”).
- Implement a mechanism to:
    - Parse and extract structured features from the text
    - Use these features as input to the prediction model
    - This component should simulate an intelligent agent interacting with users.

# 📊 Dataset
The dataset you'll be using contains data about houses and their prices. This dataset is knon as the "Ames housing" dataset, a really famous dataset used to learn about data science and machine learning.

This dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. It is a complex dataset to handle, as it contains missing data and both numerical and categorical features. The **target variable** is `"SalePrice"`, and it reports the final sale price of the house described by its realated explanatory variables.

You can get an idea of the dataset and how to process it in [this link](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html). And you can learn more about it on it's [original paper](https://jse.amstat.org/v19n3/decock.pdf) and on [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

>[!NOTE]
> This dataset is located in the `data/raw` directory. It is stored in a comma separated value (CSV) file `house_prices.txt`. The dataset contains missing values. The character `"?"` is used as a missing value marker. You can find a detailed description of all the variables in the file `data/external/data_description.txt`.

The main variables are:

- **SalePrice** – The property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass** – The building class
- **MSZoning** – The general zoning classification
- **LotFrontage** – Linear feet of street connected to property
- **LotArea** – Lot size in square feet
- **Street** – Type of road access
- **Alley** – Type of alley access
- **LotShape** – General shape of property
- **LandContour** – Flatness of the property
- **Utilities** – Type of utilities available
- **LotConfig** – Lot configuration
- **LandSlope** – Slope of property
- **Neighborhood** – Physical locations within Ames city limits
- **Condition1** – Proximity to main road or railroad
- **Condition2** – Proximity to main road or railroad (if a second is present)
- **BldgType** – Type of dwelling
- **HouseStyle** – Style of dwelling
- **OverallQual** – Overall material and finish quality
- **OverallCond** – Overall condition rating
- **YearBuilt** – Original construction date
- **YearRemodAdd** – Remodel date
- **RoofStyle** – Type of roof
- **RoofMatl** – Roof material
- **Exterior1st** – Exterior covering on house
- **Exterior2nd** – Exterior covering on house (if more than one material)
- **MasVnrType** – Masonry veneer type
- **MasVnrArea** – Masonry veneer area in square feet
- **ExterQual** – Exterior material quality
- **ExterCond** – Present condition of the material on the exterior
- **Foundation** – Type of foundation
- **BsmtQual** – Height of the basement
- **BsmtCond** – General condition of the basement
- **BsmtExposure** – Walkout or garden level basement walls
- **BsmtFinType1** – Quality of basement finished area
- **BsmtFinSF1** – Type 1 finished square feet
- **BsmtFinType2** – Quality of second finished area (if present)
- **BsmtFinSF2** – Type 2 finished square feet
- **BsmtUnfSF** – Unfinished square feet of basement area
- **TotalBsmtSF** – Total square feet of basement area
- **Heating** – Type of heating
- **HeatingQC** – Heating quality and condition
- **CentralAir** – Central air conditioning
- **Electrical** – Electrical system
- **1stFlrSF** – First floor square feet
- **2ndFlrSF** – Second floor square feet
- **LowQualFinSF** – Low quality finished square feet (all floors)
- **GrLivArea** – Above grade (ground) living area square feet
- **BsmtFullBath** – Basement full bathrooms
- **BsmtHalfBath** – Basement half bathrooms
- **FullBath** – Full bathrooms above grade
- **HalfBath** – Half baths above grade
- **Bedroom** – Number of bedrooms above basement level
- **Kitchen** – Number of kitchens
- **KitchenQual** – Kitchen quality
- **TotRmsAbvGrd** – Total rooms above grade (does not include bathrooms)
- **Functional** – Home functionality rating
- **Fireplaces** – Number of fireplaces
- **FireplaceQu** – Fireplace quality
- **GarageType** – Garage location
- **GarageYrBlt** – Year garage was built
- **GarageFinish** – Interior finish of the garage
- **GarageCars** – Size of garage in car capacity
- **GarageArea** – Size of garage in square feet
- **GarageQual** – Garage quality
- **GarageCond** – Garage condition
- **PavedDrive** – Paved driveway
- **WoodDeckSF** – Wood deck area in square feet
- **OpenPorchSF** – Open porch area in square feet
- **EnclosedPorch** – Enclosed porch area in square feet
- **3SsnPorch** – Three season porch area in square feet
- **ScreenPorch** – Screen porch area in square feet
- **PoolArea** – Pool area in square feet
- **PoolQC** – Pool quality
- **Fence** – Fence quality
- **MiscFeature** – Miscellaneous feature not covered in other categories
- **MiscVal** – Value of miscellaneous feature (in dollars)
- **MoSold** – Month sold
- **YrSold** – Year sold
- **SaleType** – Type of sale
- **SaleCondition** – Condition of sale

# 🛠️ Technology stack

The table below shows the required technology stack to implement the solution. You should use the following tools to solve the proposed problem, with special emphasis in the use of `uv` and `docker` to be able to run the complete solution without needing to install any aditional dependencies. 

>[!NOTE]
> The repository already includes the basic `uv` project configuration in `pyproject.toml` and the expected Python version in `.python-version`. To create the same local environment with the same declared packages, run:
>```bash
>uv sync
>```
<br />

| Technology | Description & Use |
|-----------|------------------|
| **uv** | Fast Python package manager and environment tool to manage dependencies and virtual environments efficiently. |
| **Python** | Core programming language used for data processing, model training, and backend development. |
| **scikit-learn** | Library for building and training the Machine Learning model, including preprocessing pipelines and evaluation. |
| **FastAPI** | High-performance web framework to build the backend API that serves model predictions. |
| **Pydantic** | Data validation and parsing library used within FastAPI to define and validate API inputs and outputs. |
| **Docker** | Containerization tool to package the application and ensure consistent execution across environments. Use `docker-compose` to orchestrate multiple services (backend, frontend, database) in a single configuration. |
| **Frontend (suggestion React)** | Web interface where users input house descriptions and view predictions and explanations. |
| **LLM / NLP layer** | Component to parse natural language descriptions into structured features (can be rule-based or LLM-powered). |
| **SHAP / Feature Importance tools** | Used to provide explainability by highlighting which features influence predictions. |
| **PostgreSQL / SQLite (optional)** | Database to store predictions, user inputs, logs, or model metadata. |
| **GitHub Actions (optional)** | CI/CD tool to automate testing, building, and deployment of the project. |
| **AI assistants (Codex, Claude Code, Copilot, etc.)** | The use of AI tools is encouraged. Feel free to use any tool to help you build your solution as long as you understand what you are implementing. |

# ✅ Evaluation criteria

| Criterion | Description |
|----------|-------------|
| **EDA (Exploratory Data Analysis)** | Quality and depth of the data analysis. Includes understanding of features, handling of missing values, identification of patterns, and clear visualizations or insights derived from the dataset. |
| **Model Training & Evaluation** | Performance and robustness of the model. Includes appropriate preprocessing, feature engineering, model selection, evaluation methodology, and justification of decisions. Generalization ability and interpretability are also considered. |
| **Model Packaging & API** | Quality of the deployed model and API. Includes proper packaging of the model, clean and functional API endpoints, input validation, error handling, and ease of use for inference. |
| **Frontend Application** | Usability and functionality of the user interface. Includes the ability to input house data (via natural language), display predictions clearly, and present explanations in an understandable way. |
| **Docker & System Integration** | Completeness of the production setup. Includes containerization of the system using Docker, orchestration with Docker Compose, and seamless integration of all components (model, API, frontend, etc.). |

# 📦 Delivery

To submit your solution, follow these instructions:

You must **fork** the provided Git repository and work on your own copy. All development and final deliverables must be included in your forked repository.

Your repository must include:

- Backend and frontend source code, fully integrated and functional.
- The notebook or script containing the EDA, pipeline creation and model traning and evaluation.
- A working docker-compose.yml file located at the root of the project.
- A SOLUTION.md file containing:
    - Clear instructions to build and run the solution.
    - A description of your technical decisions and their justification.
    - A list of aspects not implemented, along with explanations.
    - Any other relevant considerations about your approach.

Before submitting, ensure that:
- Running docker compose up starts the complete system (backend, frontend, and any additional services).
- All services are properly connected and the API endpoints are accessible.
- The application works end-to-end: from user input to prediction and explanation.

# 💡 Tips

- **Start by understanding the dataset.** Before training any model, explore the variables, check missing values, inspect distributions, and look for relationships with the target variable.

- **Create your own train/test split.** Divide the dataset into training and test sets before training the model. This is necessary to evaluate how well your solution generalizes to unseen data. Do not use the test set to make modeling decisions. Use only the training set for preprocessing, feature engineering, and model selection.

- **Use `scikit-learn` pipelines.** Pipelines help you apply the exact same preprocessing steps during both training and prediction, reducing errors and preventing inconsistencies between development and production. Operations such as imputing missing values, scaling, or encoding categorical variables must be learned from the training set and then applied to new data through the pipeline.

- **Make the model easy to reuse.** Save or serialize the trained pipeline so it can be loaded directly by the backend without retraining.

- **Design the API around real usage.** The backend should receive input data, validate it, run inference with the trained model, and return both the prediction and an explanation in a clear format.

- **Use Docker early.** Do not leave containerization until the end. Building and testing the project with Docker from the beginning will save time later. Before submitting, ensure `docker-compose` starts the full solution correctly and that all services are accessible.

- **Use AI tools as assistants (encouraged).** You are encouraged to use AI tools (e.g., ChatGPT, Copilot, etc.) to support your development process. These tools can help you explore ideas, debug issues, generate boilerplate code, or understand concepts. However, you are responsible for understanding, validating, and adapting any generated solution, as well as being able to explain your technical decisions.
