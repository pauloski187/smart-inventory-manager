# Smart Inventory Manager API Documentation

## Overview
This API provides endpoints for managing inventory, sales, suppliers, and user authentication for small businesses.

## Authentication
All endpoints except `/auth/token` require authentication via JWT token.

## Endpoints

### Authentication
- `POST /auth/token` - Login and get access token

### Products
- `GET /products` - List all products
- `POST /products` - Create a new product
- `PUT /products/{id}` - Update a product
- `DELETE /products/{id}` - Delete a product

### Suppliers
- `GET /suppliers` - List all suppliers
- `POST /suppliers` - Create a new supplier
- `PUT /suppliers/{id}` - Update a supplier
- `DELETE /suppliers/{id}` - Delete a supplier

### Sales
- `GET /sales` - List all sales
- `POST /sales` - Record a new sale

### Reports
- `GET /reports/inventory-summary` - Get inventory summary
- `GET /reports/sales-report` - Get sales report

## Frontend Integration
The Loveable AI frontend should connect to these endpoints using standard HTTP requests with Authorization headers for authenticated endpoints.