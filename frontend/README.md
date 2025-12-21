# Frontend Integration Guide

This directory is a placeholder for the frontend application that will be built using Loveable AI.

## Integration with Backend API

The frontend should be configured to communicate with the backend API running on `http://localhost:8000` (development) or your production URL.

### Key Integration Points

1. **API Base URL**: Set to the backend server URL
2. **Authentication**: Implement JWT token handling
3. **Endpoints**: Map frontend components to backend API endpoints

### Suggested Frontend Structure

- **Authentication Pages**: Login, register
- **Dashboard**: Overview with key metrics
- **Products Page**: List, add, edit, delete products
- **Inventory Page**: Stock levels, alerts
- **Sales Page**: Record sales, view history
- **Suppliers Page**: Manage suppliers
- **Reports Page**: Analytics and reports

### API Consumption Examples

```javascript
// Login
const response = await fetch('/auth/token', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    username: 'user',
    password: 'pass'
  })
});

// Get products (authenticated)
const products = await fetch('/products', {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### CORS Configuration

If the frontend is served from a different domain, ensure the backend is configured for CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development Workflow

1. Develop backend API first
2. Use Loveable AI to generate frontend components
3. Test integration between frontend and backend
4. Iterate on both as needed

## Deployment

Deploy the frontend separately from the backend, ensuring the API base URL is correctly configured for production.