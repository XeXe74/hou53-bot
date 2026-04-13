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
    HOU53-bot will help you find a fair price for any house based on your description! The problem is, we just have the data, but HOU53-bot has yet to learn about it.
    <br /><br />
    Keep on reading to learn more about this exciting challenge! 
    <br />
  </p>

![GitHub License](https://img.shields.io/github/license/Yagouus/hou53-bot)
![GitHub Release](https://img.shields.io/github/v/release/Yagouus/hou53-bot)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Yagouus/hou53-bot)

</div>


>[!WARNING] 
> **Disclaimer**: This is an educational purpose repo used as a challenge for students in NTT Data (A Coruña office). The original challenge and dataset are extracted from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Feel free to use this for educational purposes just by crediting this original source and Kaggle. For any inquire reach out to: yago.fontenlaseco@nttdata.com or angel.fragavarela@nttdata.com.


# 📖 About the challenge

The **objective of this challenge** is to tackle a real-world Machine Learning (ML) project end-to-end. You will go beyond building a model and experience the full lifecycle of an ML system: from data ingestion (ETL) and exploratory data analysis (EDA), to model training and evaluation, and finally deploying the model into production. This includes creating APIs, ensuring prediction explainability, and developing a fully functional front-end for end users.

Throughout the challenge, you will apply your knowledge of data science, machine learning, software architecture, and intelligent systems to build a complete solution that predicts house prices. The goal is to help users assess whether a property is fairly priced or estimate the value of their own home when selling.

>[!NOTE] 
> HOU53-bot is your personal house valuation assistant. It helps you estimate the price of your dream home and decide whether it’s worth buying. Trained on a rich dataset of real estate properties, HOU53-bot learns patterns between house features and their market prices. Through a simple web interface, users can provide a textual description of a property, and the system will generate a price estimation along with insights to support the decision-making process.

<picture>
      <img src="wiki/img/cover.png" alt="logo" width="800">
</picture>

The solution to the proposed problem is **open-ended**, and there are multiple ways to approach it. Feel free to explore one or all of the alternatives that you find interesting, as long as they are coherent and you can properly motivate and justify their use.

To participate, you will need to create an account on Kaggle, an online platform where datasets and problems based on those datasets are published in the form of “competitions.” This allows anyone to take part and propose the best possible solution, with many competitions offering monetary prizes for the top solutions.

For this challenge, you will work with a dataset designed for learning purposes. In the future, you will also be able to participate in official challenges hosted on the platform.

## ☝🏼 Requirements

To successfully complete this challenge, your solution must include the following components:

🧪 1. Data Analysis & Modeling
- Perform an exploratory data analysis (EDA) of the dataset to understand its structure, features, and potential issues.
- Design and implement a data preprocessing pipeline (feature engineering, handling missing values, encoding, etc.).
- Train and evaluate at least one Machine Learning model capable of predicting house prices.
Justify your modeling choices and evaluation strategy.

🚀 2. Model Deployment
- Package your trained model so it can be used in a production environment.
- Ensure the model can be loaded and used for inference independently of notebooks.

🔌 3. API Development
- Develop a backend API (e.g., REST API) that exposes an endpoint to:
- Receive input data
- Return predicted house prices
- The API should be robust, handling invalid or incomplete inputs gracefully.

🧠 4. Explainability
- Incorporate model explainability into your system.
- Provide feature importance or similar interpretability methods to explain predictions.
- Explanations should be understandable to non-technical users.

🌐 5. Frontend Application
- Develop a web-based user interface that allows users to:
- Input house information
- Receive a predicted price
- View an explanation of the prediction
- The interface should be intuitive and user-friendly.

💬 6. Natural Language Input (Agentic Component)
- The system must accept natural language descriptions of a house (e.g., “A 3-bedroom house with a large garden in a suburban area…”).
- Implement a mechanism to:
    - Parse and extract structured features from the text
    - Use these features as input to the prediction model
    - This component should simulate an intelligent agent interacting with users.

🧩 7. End-to-End Integration
- Ensure all components (model, API, frontend, and NLP parsing) are fully integrated into a working system.
- The final product should allow a user to go from text input → price prediction → explanation seamlessly.

