from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_sales():
    return {"sales": []}

@router.post("/")
def create_sale():
    return {"message": "Sale recorded"}

# Add more endpoints