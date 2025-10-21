# Machine Learning Lifecycle and MLOps: Comprehensive Notes

## 1. Feature Engineering

- Feature engineering transforms raw data into meaningful inputs for ML models.
- Features can be used directly or engineered (e.g., average expenditure per customer).
- Goal: enhance model performance by selecting the most informative features.
- Techniques:
  - Domain knowledge
  - Correlation analysis
  - Feature importance
  - Univariate selection
  - PCA (Principal Component Analysis)
  - RFE (Recursive Feature Elimination)
- Tools:
  - Feature stores (Feast, Hopsworks)
  - Data version control (DVC)

---

## 2. Experiment Tracking

- ML experiments involve varying models, hyperparameters, data versions, and environments.
- Tracking is essential for:
  - Reproducibility
  - Collaboration
  - Comparison
  - Reporting
- Methods:
  - Manual (e.g., Excel)
  - Custom platforms
  - Modern tools (MLflow, ClearML, Weights & Biases)
- Example:
  - First run: 1 hidden layer, 1,000 images
  - Second run: 2 hidden layers, 2,000 images

---

## 3. CI/CD Pipeline

- CI/CD automates development and deployment.
- **CI**: frequent code integration with automated testing.
- **CD**: automated release of validated code.
- Analogy: testing and launching a new recipe.
- Tools: Jenkins, GitLab, GitHub Actions

---

## 4. Runtime Environments and Containerization

- Development and production environments differ (e.g., Python versions).
- Containers (Docker) package code with dependencies for consistency.
- Kubernetes orchestrates and scales containers.
- Benefits:
  - Portability
  - Consistency
  - Fast startup
  - Easier maintenance

---

## 5. Deployment Architecture

- **Monolithic**: all services bundled; hard to scale and fragile.
- **Microservices**: independent services; scalable and fault-tolerant.
- ML models are deployed as microservices.
- **Inferencing**: making predictions on new data.
- **APIs**: enable service communication; standardized input/output.

---

## 6. Deployment Strategies

### Basic Deployment
- Replace old model with new.
- All traffic goes to new model.
- Easy, low resource, high risk.

### Shadow Deployment
- Send input to both models.
- Only old model used in production.
- No risk, high resource usage.

### Canary Deployment
- New model used for small traffic subset.
- Moderate risk and resource usage.
- Easier rollback.

---

## 7. Automation and Scaling

- ML lifecycle is iterative; automation speeds up experimentation.
- Scaling is essential for large data and high request volume.

### Design Phase
- Manual but can be templated.
- Automate data acquisition and quality checks.

### Development Phase
- Feature stores save time.
- Experiment tracking ensures reproducibility.

### Deployment Phase
- Containers enable scalable runtime.
- CI/CD supports fast iteration.
- Microservices architecture supports modular scaling.

---

## 8. Monitoring ML Models

- Monitoring ensures models perform reliably post-deployment.

### Statistical Monitoring
- Input/output data (e.g., churn probabilities).
- Detects prediction drift.

### Computational Monitoring
- System metrics (e.g., requests, CPU usage).
- Ensures infrastructure stability.

### Feedback Loop
- Compare predictions to ground truth.
- Identify degradation and retraining needs.

---

## 9. Retraining ML Models

- Data changes over time; retraining adapts models to new patterns.

### Types of Drift
- **Data Drift**: input distribution changes.
- **Concept Drift**: input-output relationship changes.

### Retraining Frequency
- Depends on:
  - Business environment
  - Expert insight
  - Retraining cost
  - Performance thresholds

### Retraining Strategies
- Separate models (old vs. new data)
- Combined data model
- Automatic retraining based on drift detection

---

## 10. MLOps Maturity Levels

### Level 1: Manual Processes
- No automation
- Isolated teams
- No traceability

### Level 2: Automated Development
- Feature stores, CI pipelines
- Manual deployment
- Partial monitoring and collaboration

### Level 3: Full Automation
- CI/CD for development and deployment
- Close collaboration
- Monitoring and auto-retraining

---

## 11. MLOps Tools

### Feature Stores
- **Feast**: open-source, self-managed
- **Hopsworks**: open-source, integrated platform

### Experiment Tracking
- **MLflow**: development lifecycle
- **ClearML**: development + deployment
- **Weights & Biases**: visualization-focused

### Containerization
- **Docker**: packaging
- **Kubernetes**: orchestration
- **Cloud-native**: AWS, Azure, GCP

### CI/CD
- **Jenkins**: open-source automation
- **GitLab**: proprietary with built-in CI/CD

### Monitoring
- **Fiddler**: model performance
- **Great Expectations**: data quality

### Full Lifecycle Platforms
- **AWS SageMaker**
- **Azure ML**
- **Google Cloud AI Platform**

---


