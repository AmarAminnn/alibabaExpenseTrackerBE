from fastapi import FastAPI, Depends
import models
from models import Expenses
from database import engine, SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@app.get("/expenses")
async def read_all(db: db_dependency):
    return db.query(Expenses).all()