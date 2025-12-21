from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_suppliers():
    return {"suppliers": []}

@router.post("/")
def create_supplier():
    return {"message": "Supplier created"}

# Add more endpoints for CRUD