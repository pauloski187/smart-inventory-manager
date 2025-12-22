# Smart Inventory Manager

A comprehensive inventory management system for small businesses with real-time tracking, analytics, and demand forecasting capabilities.

## Project Structure

```
smart-inventory-manager/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # FastAPI app initialization
│   │   ├── models/              # Database models
│   │   │   ├── __init__.py
│   │   │   ├── product.py       # Product model
│   │   │   ├── supplier.py      # Supplier model
│   │   │   ├── sale.py          # Sale model
│   │   │   └── user.py          # User model
│   │   ├── routes/              # API routes
│   │   │   ├── __init__.py
│   │   │   ├── products.py      # Product endpoints
│   │   │   ├── suppliers.py     # Supplier endpoints
│   │   │   ├── sales.py         # Sales endpoints
│   │   │   ├── auth.py          # Authentication endpoints
│   │   │   └── reports.py       # Reports endpoints
│   │   ├── controllers/         # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── inventory_service.py
│   │   │   └── auth_service.py
│   │   ├── middleware/          # Middleware
│   │   │   ├── __init__.py
│   │   │   └── auth_middleware.py
│   │   └── config/              # Configuration
│   │       ├── __init__.py
│   │       └── settings.py      # App settings
│   ├── docs/
│   │   └── api_docs.md          # API documentation
│   ├── tests/                   # Unit tests
│   │   └── __init__.py
│   ├── main.py                  # Application entry point
│   ├── requirements.txt         # Python dependencies
│   └── .env.example             # Environment variables template
├── frontend/                    # Frontend placeholder (built with Loveable AI)
│   └── README.md                # Frontend integration guide
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Backend Setup

1. Navigate to the backend directory:
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