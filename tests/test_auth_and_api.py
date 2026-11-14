import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_admin_seed_login():
    response = client.post("/api/auth/login", json={
        "email": "admin@health.ai",
        "password": "AdminPass123!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["role"] == "admin"

def test_user_registration_and_login():
    # Unique email
    import time
    test_email = f"test_{int(time.time())}@example.com"
    reg_response = client.post("/api/auth/register", json={
        "email": test_email,
        "password": "Password123!",
        "full_name": "Test User"
    })
    assert reg_response.status_code == 201

    login_response = client.post("/api/auth/login", json={
        "email": test_email,
        "password": "Password123!"
    })
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()

def test_prediction_api_endpoint():
    response = client.post("/api/predict", json={
        "age": 45,
        "sex": "male",
        "bmi": 31.2,
        "children": 1,
        "smoker": "yes",
        "region": "northeast"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["estimated_annual_cost"] > 10000
    assert "shap_contributions" in data
    assert len(data["shap_contributions"]) > 0
    assert data["prediction_id"] is not None

def test_history_api_and_pdf():
    # 1. Fetch history
    history_res = client.get("/api/history?limit=10")
    assert history_res.status_code == 200
    records = history_res.json()
    assert len(records) > 0
    
    first_id = records[0]["id"]
    # 2. Fetch single detail
    detail_res = client.get(f"/api/history/{first_id}")
    assert detail_res.status_code == 200
    
    # 3. Stream PDF
    pdf_res = client.get(f"/api/history/{first_id}/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000  # Non-empty PDF binary

def test_analytics_endpoints():
    dash_res = client.get("/api/analytics/dashboard")
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert "avg_predicted_cost" in dash_data
    assert "cost_distribution" in dash_data

    benchmarks_res = client.get("/api/analytics/benchmarks")
    assert benchmarks_res.status_code == 200
    benchmarks_data = benchmarks_res.json()
    assert len(benchmarks_data) >= 5

    feat_res = client.get("/api/analytics/feature-importance")
    assert feat_res.status_code == 200
    assert len(feat_res.json()) > 0

def test_auth_me_and_session_validation():
    # Login as admin to get token
    login_res = client.post("/api/auth/login", json={
        "email": "admin@health.ai",
        "password": "AdminPass123!"
    })
    token = login_res.json()["access_token"]
    
    # Authorized call
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "admin@health.ai"
    assert me_res.json()["role"] == "admin"
    
    # Unauthorized call
    unauth_res = client.get("/api/auth/me")
    assert unauth_res.status_code == 401

def test_forgot_and_reset_password_flow():
    import time
    user_email = f"reset_test_{int(time.time())}@example.com"
    # 1. Register user
    reg = client.post("/api/auth/register", json={
        "email": user_email,
        "password": "InitialPassword123!",
        "full_name": "Reset Test User"
    })
    assert reg.status_code == 201

    # 2. Request forgot password
    forgot_res = client.post("/api/auth/forgot-password", json={"email": user_email})
    assert forgot_res.status_code == 200
    reset_token = forgot_res.json()["reset_token"]
    assert reset_token is not None

    # 3. Reset password using token
    new_pass = "BrandNewSecretPassword999!"
    reset_res = client.post("/api/auth/reset-password", json={
        "token": reset_token,
        "new_password": new_pass
    })
    assert reset_res.status_code == 200

    # 4. Old password should fail
    old_fail = client.post("/api/auth/login", json={
        "email": user_email,
        "password": "InitialPassword123!"
    })
    assert old_fail.status_code == 401

    # 5. New password should succeed
    new_ok = client.post("/api/auth/login", json={
        "email": user_email,
        "password": new_pass
    })
    assert new_ok.status_code == 200
    assert "access_token" in new_ok.json()

def test_auth_logout():
    logout_res = client.post("/api/auth/logout")
    assert logout_res.status_code == 200

def test_plans_comparison_api():
    res = client.get("/api/plans")
    assert res.status_code == 200
    plans = res.json()
    assert len(plans) >= 4
    assert any("HMO" in p["category"] for p in plans)
    assert any("PPO" in p["category"] for p in plans)

    rec_res = client.post("/api/plans/recommend", json={
        "predicted_annual_cost": 28000.0,
        "risk_tier": "Critical Risk",
        "age": 55,
        "smoker": "yes"
    })
    assert rec_res.status_code == 200
    assert "recommended_plan" in rec_res.json()

def test_ai_assistant_chat_api():
    chat_res = client.post("/api/assistant/chat", json={
        "message": "What is the difference between deductible and coinsurance?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert "deductible" in data["reply"].lower()
    assert len(data["suggested_questions"]) > 0

def test_what_if_simulation_api():
    sim_res = client.post("/api/predict/simulate", json={
        "baseline": {
            "age": 45, "sex": "male", "bmi": 32.5, "children": 1, "smoker": "yes", "region": "southeast"
        },
        "modified": {
            "age": 45, "sex": "male", "bmi": 24.5, "children": 1, "smoker": "no", "region": "southeast"
        }
    })
    assert sim_res.status_code == 200
    data = sim_res.json()
    assert data["comparison"]["delta_annual_cost"] < 0
    assert data["comparison"]["is_savings"] is True
    assert len(data["comparison"]["key_drivers"]) > 0

def test_history_csv_export_api():
    csv_res = client.get("/api/history/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "Predicted Annual Charges" in csv_res.text

def test_admin_user_management_api():
    # Admin login
    login_res = client.post("/api/auth/login", json={
        "email": "admin@health.ai",
        "password": "AdminPass123!"
    })
    token = login_res.json()["access_token"]
    users_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert users_res.status_code == 200
    users = users_res.json()
    assert len(users) >= 1
    assert any(u["email"] == "admin@health.ai" for u in users)


