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

# Post-Deployment: Monitoring, Retraining, Maturity, and Tooling in MLOps

## 1. Monitoring Machine Learning Models

- Once deployed, models must be monitored to ensure they perform reliably on new, unseen data.
- Monitoring helps detect performance degradation, infrastructure issues, and data drift.

### Types of Monitoring

- **Statistical Monitoring**:
  - Focuses on input data and model predictions.
  - Example: tracking churn probability outputs.
  - Detects shifts in data distribution or prediction quality.

- **Computational Monitoring**:
  - Focuses on system-level metrics.
  - Includes request volume, network usage, CPU/memory consumption.
  - Ensures infrastructure stability and responsiveness.

### Analogy

- Like monitoring a kitchen:
  - **Computational**: check appliances, electricity, staff activity.
  - **Statistical**: inspect ingredient quality and dish taste.

### Feedback Loop

- Ground truth (actual outcomes) becomes available over time.
- Comparing predictions to ground truth reveals:
  - Accuracy trends
  - Model degradation
  - Biases or failure patterns
- Enables retraining and continuous improvement.

---

## 2. Retraining Machine Learning Models

- Retraining adapts models to new data patterns and maintains performance over time.

### Why Retraining Is Needed

- Data changes due to evolving user behavior, market conditions, or external factors.
- Models trained on outdated data may become inaccurate.

### Types of Drift

- **Data Drift**:
  - Change in input data distribution.
  - Example: new customer demographics.

- **Concept Drift**:
  - Change in relationship between input and target.
  - Example: same customer profile now leads to non-churn instead of churn.

### Retraining Frequency

- Depends on:
  - Business environment volatility
  - Expert insight
  - Cost of retraining (compute, time)
  - Performance thresholds (e.g., accuracy > 90%)
  - Rate of model degradation

### Retraining Strategies

- **Separate Models**:
  - Train new model on new data only.
  - Maintain old model for reference or fallback.

- **Combined Data**:
  - Merge old and new data to train a unified model.
  - Choice depends on domain sensitivity and cost.

### Automatic Retraining

- Triggered by:
  - Data volume thresholds
  - Drift detection (e.g., average customer age shifts)
- Requires mature infrastructure and monitoring systems.

---

## 3. MLOps Maturity Levels

- Maturity reflects automation, collaboration, and monitoring across ML workflows.

### Level 1: Manual Processes

- No automation; all tasks are manual.
- Teams work in silos.
- No traceability or reproducibility.
- Common starting point for new ML teams.

### Level 2: Automated Development

- Development is automated (feature stores, CI pipelines).
- Deployment is manual.
- Partial collaboration and monitoring.
- Models are reproducible during development.

### Level 3: Full Automation

- CI/CD automates development and deployment.
- Strong collaboration across roles.
- Active monitoring and auto-retraining.
- Scalable and production-ready ML systems.

---

## 4. MLOps Tools by Lifecycle Component

### Feature Stores

- **Feast**:
  - Open-source, self-managed
  - Flexible but requires manual setup

- **Hopsworks**:
  - Open-source, part of Hopsworks platform
  - Best used with other Hopsworks tools

### Experiment Tracking

- **MLflow**:
  - Tracks parameters, metrics, artifacts
  - Focused on development

- **ClearML**:
  - Covers development and deployment
  - Offers orchestration tools

- **Weights & Biases (W&B)**:
  - Specializes in experiment visualization
  - Strong dashboarding and collaboration

### Containerization

- **Docker**:
  - Packages code with dependencies
  - Ensures consistent runtime

- **Kubernetes**:
  - Orchestrates and scales containers
  - Enables automated deployment

- **Cloud-native options**:
  - AWS, Azure, GCP offer integrated container services

### CI/CD

- **Jenkins**:
  - Open-source automation server
  - Highly customizable

- **GitLab**:
  - Proprietary platform with built-in CI/CD
  - Supports collaborative development

### Monitoring

- **Fiddler**:
  - Model performance monitoring
  - Tracks prediction accuracy and drift

- **Great Expectations**:
  - Data quality monitoring
  - Validates schema, missing values, distribution shifts

### End-to-End Platforms

- **AWS SageMaker**
- **Azure Machine Learning**
- **Google Cloud AI Platform**

Each provides:
- Data exploration
- Feature engineering
- Model training
- Deployment
- Monitoring

---


