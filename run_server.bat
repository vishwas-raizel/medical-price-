@echo off
title HealthRisk AI - Production Platform
echo ===============================================================================
echo     HealthRisk AI - Medical Insurance Predictive Risk Intelligence System
echo ===============================================================================
echo.
echo Starting Production Server on http://localhost:8000 ...
echo.
echo - Web Dashboard:              http://localhost:8000/
echo - Interactive Swagger API:    http://localhost:8000/docs
echo - Alternative Redoc API:      http://localhost:8000/redoc
echo.
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
pause


python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload