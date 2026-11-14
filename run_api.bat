@echo off
echo =====================================================================
echo  Starting Medical Insurance Price Prediction REST API Server
echo =====================================================================
echo.
echo Launching FastAPI Server on http://127.0.0.1:8000
echo Interactive API Documentation: http://127.0.0.1:8000/docs
echo.
python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
pause
