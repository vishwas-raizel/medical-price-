import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.database.session import engine, Base, SessionLocal
from backend.app.models.user import User
from backend.app.services.auth_service import get_password_hash
from backend.app.routes import auth, predict, history, analytics, admin, plans, assistant

# 1. Initialize Database Schema
Base.metadata.create_all(bind=engine)

# 2. Seed Default Admin User if Database is New
def seed_initial_data():
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@health.ai").first()
        if not admin_user:
            admin_user = User(
                email="admin@health.ai",
                hashed_password=get_password_hash("AdminPass123!"),
                full_name="Platform Lead Administrator",
                role="admin",
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print("Default Administrator created: admin@health.ai / AdminPass123!")
    except Exception as e:
        db.rollback()
        print(f"Database seed note: {e}")
    finally:
        db.close()

seed_initial_data()

# 3. Create FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Enterprise Machine Learning Healthcare Underwriting, Risk Intelligence & Explainable AI REST API.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 4. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include API Routers
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(predict.router, prefix=settings.API_V1_PREFIX)
app.include_router(history.router, prefix=settings.API_V1_PREFIX)
app.include_router(analytics.router, prefix=settings.API_V1_PREFIX)
app.include_router(admin.router, prefix=settings.API_V1_PREFIX)
app.include_router(plans.router, prefix=settings.API_V1_PREFIX)
app.include_router(assistant.router, prefix=settings.API_V1_PREFIX)

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "timestamp": os.getenv("CURRENT_TIME", "2026-11-14T20:00:00Z")
    }

# 6. Mount Static Frontend (React SPA) & Reports
static_dir = os.path.join(root_dir, "frontend", "public")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

reports_dir = os.path.join(root_dir, "reports")
if os.path.exists(reports_dir):
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

@app.get("/")
def serve_root():
    index_path = os.path.join(root_dir, "frontend", "public", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Medical Insurance Price Prediction API is Online",
        "docs": "/docs",
        "api_endpoints": f"{settings.API_V1_PREFIX}/predict"
    }

# Fallback for client-side routing
@app.get("/{full_path:path}")
def catch_all(full_path: str):
    # If API call or reports call not found, return 404
    if full_path.startswith("api") or full_path.startswith("reports"):
        return {"detail": "Not Found"}
    index_path = os.path.join(root_dir, "frontend", "public", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API Running", "docs": "/docs"}
