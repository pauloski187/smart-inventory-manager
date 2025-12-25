# Smart Inventory Manager

A comprehensive machine learning solution for inventory optimization and demand forecasting in e-commerce retail.

## 📋 Project Overview

This project implements an end-to-end inventory management system using machine learning to:
- Forecast product demand
- Classify inventory items using ABC analysis
- Identify dead stock
- Optimize inventory levels
- Provide actionable business insights

## 🏗️ Project Structure

```
smart-inventory-manager/
├── data/
│   ├── raw/                    # Raw data files
│   │   └── amazon_orders.csv   # Sample sales data
│   └── processed/              # Processed data and features
├── ml/
│   ├── notebooks/              # Jupyter notebooks for analysis
│   │   ├── 01_data_overview.ipynb
│   │   ├── 02_inventory_eda.ipynb
│   │   ├── 03_sales_trends.ipynb
│   │   ├── 04_feature_engineering.ipynb
│   │   ├── 05_demand_forecasting.ipynb
│   │   ├── 06_dead_stock_analysis.ipynb
│   │   ├── 07_abc_analysis.ipynb
│   │   └── 08_model_evaluation.ipynb
│   ├── scripts/                # Python scripts for automation
│   │   ├── train_forecast_model.py
│   │   ├── evaluate_model.py
│   │   └── feature_pipeline.py
│   └── models/                 # Trained models and artifacts
│       ├── demand_forecast.pkl
│       ├── abc_classifier.pkl
│       └── feature_pipeline.pkl
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Lab
- Required packages (see installation below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd smart-inventory-manager
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. **Navigate to the ML directory:**
   ```bash
   cd ml
   ```

### Usage

#### Option 1: Run Notebooks (Interactive Analysis)

1. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Execute notebooks in order:**
   - `01_data_overview.ipynb` - Initial data exploration
   - `02_inventory_eda.ipynb` - Exploratory data analysis
   - `03_sales_trends.ipynb` - Sales trend analysis
   - `04_feature_engineering.ipynb` - Feature creation
   - `05_demand_forecasting.ipynb` - Model training
   - `06_dead_stock_analysis.ipynb` - Dead stock identification
   - `07_abc_analysis.ipynb` - ABC inventory classification
   - `08_model_evaluation.ipynb` - Model performance evaluation

#### Option 2: Run Scripts (Automated Pipeline)

1. **Feature Engineering:**
   ```bash
   python scripts/feature_pipeline.py
   ```

2. **Train Model:**
   ```bash
   python scripts/train_forecast_model.py
   ```

3. **Evaluate Model:**
   ```bash
   python scripts/evaluate_model.py
   ```

## 📊 Data Description

The sample dataset (`amazon_orders.csv`) contains:
- **order_id**: Unique order identifier
- **order_date**: Date of the order
- **product_id**: Unique product identifier
- **quantity**: Quantity ordered
- **unit_price**: Price per unit

## 🔧 Components

### 1. Data Overview (`01_data_overview.ipynb`)
- Data loading and cleaning
- Basic statistics and distributions
- Initial data quality assessment

### 2. Inventory EDA (`02_inventory_eda.ipynb`)
- Product performance analysis
- Sales patterns and seasonality
- Inventory metrics calculation

### 3. Sales Trends (`03_sales_trends.ipynb`)
- Monthly and seasonal sales analysis
- Sales velocity calculations
- Trend identification

### 4. Feature Engineering (`04_feature_engineering.ipynb`)
- Time-based features (month, day, weekend indicators)
- Lag features (previous days' sales)
- Rolling statistics (moving averages, standard deviations)
- Product-specific features

### 5. Demand Forecasting (`05_demand_forecasting.ipynb`)
- Random Forest model training
- 7-day ahead demand prediction
- Feature importance analysis
- Model persistence

### 6. Dead Stock Analysis (`06_dead_stock_analysis.ipynb`)
- Identification of slow-moving inventory
- Dead stock categorization (Critical, High Risk, Medium Risk, Active)
- Financial impact assessment
- Action plan generation

### 7. ABC Analysis (`07_abc_analysis.ipynb`)
- Inventory classification (A-items: 20% items, 80% value)
- Product prioritization
- Management recommendations

### 8. Model Evaluation (`08_model_evaluation.ipynb`)
- Comprehensive performance metrics (MAE, RMSE, MAPE, R²)
- Cross-validation analysis
- Business impact assessment
- Deployment recommendations

## 📈 Key Features

### Demand Forecasting
- **7-day ahead predictions** using historical sales data
- **Multiple feature types**: time-based, lag, rolling statistics
- **Model**: Random Forest Regressor with feature selection

### Inventory Classification
- **ABC Analysis**: Pareto-based inventory categorization
- **Dead Stock Detection**: Automated identification of slow-moving items
- **Actionable Insights**: Prioritized recommendations for inventory management

### Business Impact
- **Cost Savings**: Reduced carrying costs through better forecasting
- **Stock Optimization**: Minimized stock-outs and overstock situations
- **ROI Tracking**: Quantified financial benefits

## 🎯 Performance Metrics

Typical model performance on sample data:
- **MAE**: 2.5 units
- **MAPE**: 15.0%
- **R²**: 0.75

## 🔄 Pipeline Automation

The project includes automated scripts for:
- **Feature Engineering**: `feature_pipeline.py`
- **Model Training**: `train_forecast_model.py`
- **Model Evaluation**: `evaluate_model.py`

## 📋 Business Applications

1. **Inventory Optimization**
   - Reduce carrying costs
   - Minimize stock-outs
   - Optimize reorder points

2. **Demand Planning**
   - Accurate sales forecasting
   - Seasonal trend analysis
   - Product lifecycle management

3. **ABC Classification**
   - Focus management efforts on high-value items
   - Optimize storage and handling
   - Improve supplier negotiations

4. **Dead Stock Management**
   - Identify slow-moving inventory
   - Reduce obsolescence costs
   - Improve cash flow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the notebooks for detailed documentation
- Review the code comments for implementation details

## 🔄 Future Enhancements

- [ ] Deep learning models (LSTM, Transformer)
- [ ] Real-time forecasting
- [ ] Multi-product forecasting
- [ ] External factor integration (weather, holidays, promotions)
- [ ] Automated retraining pipeline
- [ ] Web dashboard for results visualization
- [ ] API endpoints for model serving
- [ ] Integration with inventory management systems
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your actual values.

4. Run database migrations (if using Alembic):
   ```bash
   alembic upgrade head
   ```

5. Start the server:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## Frontend Integration

The frontend will be developed using Loveable AI. It should connect to the backend API endpoints.

### API Base URL
Set the base URL to `http://localhost:8000` (or your production URL).

### Authentication
- Use `POST /auth/token` to login and get JWT token
- Include the token in `Authorization: Bearer <token>` header for protected endpoints

### Available Endpoints
See `backend/docs/api_docs.md` for detailed API documentation.

### CORS
Ensure the frontend handles CORS if needed. FastAPI can be configured for CORS in the app initialization.

## Features

- **Product Management**: CRUD operations for products
- **Stock Tracking**: Monitor stock levels and low stock alerts
- **Sales Tracking**: Record and track sales
- **Supplier Management**: Manage supplier information
- **User Authentication**: Secure login system
- **Reports & Analytics**: Generate inventory and sales reports

## Database Schema

The application uses SQLAlchemy ORM with the following main tables:
- `products`: Product information and stock levels
- `suppliers`: Supplier details
- `sales`: Sales transactions
- `users`: User accounts for authentication

## Development

- Backend: Python 3.8+, FastAPI, SQLAlchemy
- Database: SQLite (default), can be changed to PostgreSQL/MySQL
- Authentication: JWT tokens
- API Documentation: Automatic Swagger UI at `/docs`

## Deployment

1. Set up a production database
2. Update environment variables
3. Run migrations
4. Deploy the FastAPI app using uvicorn or a WSGI server
5. Configure the frontend to point to the production API URL