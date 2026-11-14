import urllib.request
import json
import re

def verify_all():
    print("1. Fetching live HTML from http://127.0.0.1:8000/...")
    with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
        html = resp.read().decode("utf-8")
        assert resp.status == 200
        print(f"   HTTP Status: {resp.status}, HTML size: {len(html)} bytes")

    ids = [
        "dashAvgCost", "dashMedianCost", "dashHighRiskPct", "dashChampionR2", "dashChampionName",
        "chartChargesDist", "chartAgeCohort", "dashRecentTbody",
        "benchmarksTbody", "chartModelR2", "chartModelRmse", "chartActualVsPred", "chartResidualDist", "chartAnalyticsFeatureImp",
        "activeGalleryImg", "activeGalleryCaption", "plotGalleryButtons"
    ]

    print("\n2. Checking required DOM elements in index.html:")
    missing = 0
    for elem_id in ids:
        if f'id="{elem_id}"' in html or f"id='{elem_id}'" in html:
            print(f"   ✓ PASS: #{elem_id}")
        else:
            print(f"   ✗ FAIL: #{elem_id} NOT FOUND!")
            missing += 1

    assert missing == 0, f"{missing} elements missing in DOM"

    print("\n3. Testing Analytics API Endpoints:")
    endpoints = [
        ("/api/analytics/dashboard", ["total_predictions", "avg_predicted_cost", "median_cost", "high_risk_percentage", "cost_distribution", "age_cohort_averages"]),
        ("/api/analytics/benchmarks", None), # is list
        ("/api/analytics/feature-importance", None), # is list
        ("/api/analytics/residuals", ["bins", "counts", "mean_residual", "std_residual"]),
        ("/api/analytics/actual-vs-predicted", None), # is list
        ("/api/history?limit=5", None) # is list
    ]

    for ep, required_keys in endpoints:
        url = f"http://127.0.0.1:8000{ep}"
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict) and required_keys:
                for k in required_keys:
                    assert k in data, f"Key {k} missing in {ep}"
                print(f"   ✓ PASS: {ep} returned valid dict with all {len(required_keys)} expected keys")
            elif isinstance(data, list):
                print(f"   ✓ PASS: {ep} returned valid list with {len(data)} items")
            else:
                print(f"   ✓ PASS: {ep} returned {type(data)}")

    print("\n4. Testing Static Reports Figures Mount:")
    figures = [
        "actual_vs_predicted.png",
        "residuals_distribution.png",
        "feature_importance.png",
        "eda_correlation_heatmap.png",
        "eda_bmi_vs_charges_by_smoker.png",
        "eda_age_vs_charges.png"
    ]
    for fig in figures:
        url = f"http://127.0.0.1:8000/reports/figures/{fig}"
        with urllib.request.urlopen(url) as r:
            content_type = r.headers.get("Content-Type")
            data = r.read()
            assert r.status == 200
            assert "image/png" in content_type, f"Expected image/png, got {content_type}"
            assert len(data) > 1000, "Image byte length too small"
            print(f"   ✓ PASS: {fig} served as {content_type} ({len(data):,} bytes)")

    print("\n=== ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    verify_all()
