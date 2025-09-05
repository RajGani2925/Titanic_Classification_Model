
# 🚢 Comparative Analysis of Machine Learning Models on the Titanic Classification Dataset 🤖

This project applies multiple machine learning algorithms to the famous Titanic dataset 🛳️ in order to predict passenger survival 🎯. The dataset contains demographic and travel-related features such as age 👶👴, sex 🚹🚺, fare 💵, passenger class 🎟️, and embarkation point ⚓.

The study evaluates the performance of several classification models, including:
🔹 K-Nearest Neighbors (KNN)
🔹 Support Vector Classifier (SVC)
🔹 Random Forest 🌲
🔹 Decision Tree 🌳
🔹 Logistic Regression 📈

Each model was trained and tested on the dataset, and their accuracy scores 📊 were compared to determine their effectiveness in survival prediction.

Through this comparative analysis, the project highlights the strengths 💪 and limitations ⚠️ of different classification techniques when applied to a real-world dataset. The results provide insights into which models are better suited for structured datasets with mixed categorical and numerical features 🔢, such as the Titanic dataset.

## Introduction
The Titanic – Machine Learning from Disaster dataset 🚢 is one of the most iconic beginner-friendly datasets for classification tasks 🤖. This project focuses on building predictive models to determine whether a passenger survived the Titanic tragedy 💡 based on features such as age 👶👴, sex 🚹🚺, passenger class 🎟️, fare 💵, and embarkation point ⚓.

The workflow includes:
🔹 Data Cleaning 🧹
🔹 Exploratory Data Analysis (EDA) 🔍
🔹 Feature Engineering 🛠️
🔹 Model Training & Evaluation 📊

Several machine learning models were implemented, including:
✨ Logistic Regression 📈
✨ K-Nearest Neighbors (KNN) 👥
✨ Support Vector Classifier (SVC) 📐
✨ Decision Tree 🌳
✨ Random Forest 🌲

The main objectives of this project are:
1️⃣ Practical Learning 🎯 – to gain hands-on experience with the machine learning pipeline, from preprocessing to evaluation.
2️⃣ Comparative Analysis ⚖️ – to compare multiple classification algorithms and identify which performs best on the Titanic dataset.

This hands-on exercise serves as a strong foundation for understanding supervised learning concepts 🧠, model selection 🔄, and the importance of preprocessing 🧩 in real-world datasets.

## Titanic Dataset – Feature Explanation

1. **PassengerId 🔢**

   * Unique identifier for each passenger.
   * Not useful for prediction (just an index).

2. **Survived ⚰️➡️🙂**

   * Target variable (0 = did not survive, 1 = survived).
   * What your models are trying to predict.

3. **Pclass 🎟️**

   * Passenger’s ticket class:

     * 1 = First Class (luxury 🥂)
     * 2 = Second Class (middle 💺)
     * 3 = Third Class (economy 🚶)
   * A proxy for socio-economic status.

4. **Name 🏷️**

   * Passenger’s full name.
   * Can be used to extract **titles** (Mr, Mrs, Miss, etc.) which often help prediction.

5. **Sex 🚹🚺**

   * Gender of the passenger.
   * Very important feature (women had higher survival rates).

6. **Age 👶👦👩👴**

   * Passenger’s age in years.
   * Some values are missing (\~177 NaNs).
   * Survival rates often varied by age group (children had priority).

7. **SibSp 👨‍👩‍👧**

   * Number of siblings/spouses aboard.
   * Shows family size onboard.

8. **Parch 👶👵**

   * Number of parents/children aboard.
   * Another indicator of family size.

9. **Ticket 🎫**

   * Ticket number (categorical).
   * Not very predictive directly, but can sometimes encode group/family info.

10. **Fare 💵**

* Price paid for the ticket.
* Higher fares often linked to higher class (and higher survival chances).

11. **Cabin 🚪**

* Cabin number.
* Mostly missing (only \~204 filled).
* Can sometimes hint at passenger class and location on the ship.

12. **Embarked ⚓**

* Port of Embarkation:

  * C = Cherbourg 🇫🇷
  * Q = Queenstown 🇮🇪
  * S = Southampton 🏴
* Has 2 missing values.

---

✅ **Summary:**

* **Categorical features**: Sex, Embarked, Pclass, Name (derived titles), Ticket, Cabin
* **Numerical features**: Age, SibSp, Parch, Fare
* **Target variable**: Survived

---

## Results
The performance of different classification models was evaluated on the Titanic dataset 🚢 to predict passenger survival ⚰️➡️🙂. Each model was trained and tested after preprocessing the data, and their accuracy scores 🎯 were compared:

📈 Logistic Regression → ~80% accuracy

🌳 Decision Tree → ~77% accuracy

🌲 Random Forest → ~82% accuracy

👥 K-Nearest Neighbors (KNN) → ~78% accuracy

📐 Support Vector Classifier (SVC) → ~78% accuracy

✅ Key Insight: Ensemble models like Random Forest 🌲 achieved the highest accuracy, while Logistic Regression 📈 also performed strongly as a baseline model. Simpler models like Decision Trees 🌳 and KNN 👥 gave competitive but slightly lower results.

This comparison highlights the importance of trying multiple algorithms ⚖️ and selecting the best-performing one for the problem at hand.
## Future Works/Improvements
This project already includes strong feature engineering (e.g., extracted passenger titles 🏷️, created family size 👨‍👩‍👧) and hyperparameter tuning ⚙️. Possible next steps to expand the project include:

✨ **More Advanced Models** 🤖

🔹Implement gradient boosting models like XGBoost ⚡, LightGBM 🌟, or CatBoost 🐈

🔹Compare results with deep learning models (TensorFlow / PyTorch) 🧠

✨ **Cross-Validation & Robust Evaluation** 🔁

🔹Extend beyond a simple train-test split to k-fold cross-validation for more reliable accuracy.

✨ **Better Handling of Missing Data** 🧩

🔹Try advanced imputation methods (e.g., KNN imputer, iterative imputer) instead of simple fill techniques.

✨ **Model Interpretability** 🔍

🔹Use SHAP or LIME to explain predictions and identify the most influential features.

✨ **Deployment** 🚀

🔹Create a Streamlit/Dash web app for interactive survival prediction 🌐 Or develop a REST API with FastAPI/Flask so the model can be integrated into other systems.
## Acknowledgements
This project is heavily inspired 🙏 by the work of [Berkay Doğan](https://www.linkedin.com/in/berkaydogan-/)
 👨‍💻. His [kaggle Profile](https://www.kaggle.com/berkaydogan1)
  provided the foundation for much of the code used here. While several parts were adapted 🔧, the overall structure is almost a replica of his implementation.

The main ambition behind this project was to learn coding practices 💻 and develop intuition for machine learning workflows 🤖. I believe that studying 📚 and replicating well-written code is a valuable step in building a deeper understanding 💡 of the concepts and techniques behind data science projects 🔬.

If you want to know more this is his [Github-Profile](https://github.com/berkaydgn).
 🐙.
