# SOLUTION.md — HOU53-bot

## How to Build and Run the Solution

The complete system is containerized using Docker and orchestrated with Docker Compose. The only prerequisites are having **Docker Desktop** installed and running. No additional dependencies need to be installed manually.

To build and start all services, run the following command from the root of the repository:

```bash
docker compose up --build
```

This single command will:
1. Build the backend image (Python 3.12 + uv + FastAPI)
2. Build the frontend image (Node 20 + Vite → served via Nginx)
3. Pull and start the Ollama service
4. Automatically download the `llama3.2:3b` model (~2 GB — first run only)

Once all services are running, the application is accessible at:

- **Frontend**: http://localhost:3000
- **Backend API docs**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

> **Note:** The first startup may take several minutes due to the model download. Subsequent runs use cached Docker layers and the persisted Ollama volume, so they start significantly faster.

To stop all services:

```bash
docker compose down
```

---

## Technical Decisions and Justifications

### Data Analysis and Preprocessing

Exploratory data analysis was conducted to understand the distribution of features, identify missing values, and detect correlations with the target variable `SalePrice`. The dataset contains 79 features, many of which have missing values represented by the `?` character. Preprocessing decisions were driven by the nature of each variable: numerical features with missing values were imputed using the median, while categorical features used the most frequent value or a dedicated `"None"` category where missingness carries semantic meaning (e.g., no basement, no garage).

Feature engineering focused on capturing non-linear relationships — for example, a combined `QualityArea` feature multiplying `OverallQual` by `GrLivArea`, which consistently emerged as the strongest predictor. Categorical variables were encoded using ordinal encoding for quality-based features (where order matters) and one-hot encoding for nominal variables.

The entire preprocessing logic was encapsulated in a `scikit-learn` pipeline to ensure that the exact same transformations applied during training are reused at inference time, preventing data leakage and guaranteeing consistency between development and production.

### Model Selection

A **Lasso regression** model was selected as the primary predictive model. Lasso was chosen because it performs L1 regularization, which automatically drives irrelevant feature coefficients to zero and produces sparse, interpretable models — a desirable property given the high dimensionality of the Ames Housing dataset. The model was tuned using cross-validation to select the optimal regularization strength (`alpha`).

Lasso's linear nature also makes it compatible with SHAP (SHapley Additive exPlanations), enabling reliable and fast feature importance explanations without approximation overhead.

### Explainability

Model explainability is provided via **SHAP values**, computed using a `LinearExplainer` matched to the Lasso model. For each prediction, the top contributing features are returned alongside their SHAP importance scores. These are displayed in the frontend in a user-friendly format, showing non-technical users which aspects of the house most influenced the estimated price.

### API Design

The backend was built with **FastAPI** and exposes the following main endpoints:

- `POST /predict` — receives a natural language description, parses it into structured features via the LLM, runs the prediction pipeline, and returns the estimated price, confidence range, and SHAP explanation.
- `GET /health` — basic health check endpoint.

Input validation is handled by Pydantic models. The API handles missing or incomplete feature extractions gracefully by falling back to dataset median values, ensuring robust predictions even from partial descriptions.

### Natural Language Parsing

The natural language parsing layer uses **Ollama** running `llama3.2:3b` locally. When a user submits a textual description of a house, the backend sends a structured prompt to the LLM requesting a JSON response with the extracted feature values. The model is instructed to map natural language expressions (e.g., "excellent kitchen") to the dataset's ordinal scale values (e.g., `KitchenQual=Ex`).

This approach was chosen over a rule-based parser because it handles varied phrasing, partial descriptions, and ambiguous language far more robustly, while keeping the entire system self-contained without requiring external API keys.

### Frontend

The frontend was built with **React + Vite**, served in production via **Nginx** inside a Docker container. The interface allows users to type a natural language description and receive the predicted price, confidence interval, and a feature importance breakdown. The `VITE_API_URL` environment variable controls the backend URL, making the frontend environment-agnostic.

---

## Aspects Not Implemented

| Aspect | Reason |
|--------|--------|
| **PostgreSQL / SQLite database** | Logging predictions and user inputs was considered out of scope for this version. The system functions correctly without persistence. |
| **GitHub Actions CI/CD** | Automated testing and deployment pipelines were not implemented, as the focus was on the core ML and deployment requirements. |
| **Multiple model comparison** | Only Lasso was deployed. Ridge regression and Gradient Boosting were evaluated during experimentation but not exposed via the API. |
| **User authentication** | No login system was implemented, as it was not required by the project specification. |

---

## Additional Considerations

The system runs entirely on CPU inside Docker. The `llama3.2:3b` model was selected specifically because it is small enough to run on machines without a GPU while still producing reliable feature extraction from natural language. On machines with limited RAM (under 8 GB), Ollama may be slow to respond — increasing the `LLM_TIMEOUT` environment variable in `docker-compose.yml` can mitigate timeout errors.

The Ollama Docker image does not include `curl` or `wget`, which prevents the use of standard Docker healthchecks. The `docker-compose.yml` uses a `sleep 10` delay in the `ollama-setup` service instead of a `condition: service_healthy` dependency to work around this known upstream limitation.

The trained model and preprocessing pipeline are serialized and stored in the `data/` directory, allowing the backend to load them at startup without retraining. This ensures fast cold starts and consistent inference behavior across environments.
