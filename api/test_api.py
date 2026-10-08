"""
test_api.py - ATHENA API Testing Script
Tests all FastAPI endpoints
Author: ATHENA Project - Week 2
"""

import requests
import json
from time import sleep

# API base URL (change when deployed)
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Root Endpoint (/)")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Root endpoint working")


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Health Check (/health)")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response:\n{json.dumps(data, indent=2)}")
    
    assert response.status_code == 200
    assert "status" in data
    assert "model_loaded" in data
    
    if data["model_loaded"]:
        print("✅ Health check passed - Model loaded")
    else:
        print("⚠️  Health check passed - Model NOT loaded (train model first)")


def test_stocks():
    """Test supported stocks endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Supported Stocks (/stocks)")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stocks")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response:\n{json.dumps(data, indent=2)}")
    
    assert response.status_code == 200
    assert "stocks" in data
    assert "count" in data
    assert data["count"] > 0
    
    print(f"✅ Found {data['count']} supported stocks")


def test_predict(ticker="AAPL"):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print(f"TEST 4: Stock Prediction (/predict) - {ticker}")
    print("="*60)
    
    # Check if model is loaded first
    health_response = requests.get(f"{BASE_URL}/health")
    if not health_response.json()["model_loaded"]:
        print("⚠️  Skipping prediction test - Model not loaded")
        print("   Please train the model first: python train.py")
        return
    
    payload = {"ticker": ticker}
    print(f"Request payload: {payload}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n📊 Prediction Results:")
        print(f"   Ticker: {data['ticker']}")
        print(f"   Current Price: ${data['current_price']:.2f}")
        print(f"   Predicted Price: ${data['predicted_price']:.2f}")
        print(f"   Direction: {data['direction']}")
        print(f"   Change: {data['change_percent']:+.2f}%")
        print(f"   Confidence: {data['confidence']*100:.1f}%")
        
        print("\n🔑 Top Features:")
        for i, feature in enumerate(data['top_features'][:3], 1):
            print(f"   {i}. {feature['name']}: {feature['impact']:+.4f}")
        
        print("\n📅 Important Days:")
        for i, day in enumerate(data['important_days'][:3], 1):
            print(f"   {i}. {day['days_ago']} days ago (importance: {day['importance']:.3f})")
        
        print("\n✅ Prediction endpoint working")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")


def test_multiple_stocks():
    """Test predictions for multiple stocks"""
    print("\n" + "="*60)
    print("TEST 5: Multiple Stock Predictions")
    print("="*60)
    
    # Check if model is loaded first
    health_response = requests.get(f"{BASE_URL}/health")
    if not health_response.json()["model_loaded"]:
        print("⚠️  Skipping multiple predictions test - Model not loaded")
        return
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = []
    
    for ticker in tickers:
        print(f"\n📊 Testing {ticker}...")
        payload = {"ticker": ticker}
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            results.append({
                "ticker": data["ticker"],
                "predicted": data["predicted_price"],
                "direction": data["direction"],
                "confidence": data["confidence"]
            })
            print(f"   ✅ {ticker}: ${data['predicted_price']:.2f} ({data['direction']})")
        else:
            print(f"   ❌ {ticker}: Error {response.status_code}")
        
        # Small delay to avoid rate limiting
        sleep(1)
    
    print("\n" + "="*60)
    print("📊 SUMMARY OF PREDICTIONS")
    print("="*60)
    for r in results:
        print(f"{r['ticker']}: ${r['predicted']:.2f} {r['direction']} (Confidence: {r['confidence']*100:.1f}%)")
    
    print(f"\n✅ Tested {len(results)}/{len(tickers)} stocks successfully")


def test_error_handling():
    """Test API error handling"""
    print("\n" + "="*60)
    print("TEST 6: Error Handling")
    print("="*60)
    
    # Test invalid ticker
    print("\n1. Testing invalid ticker...")
    payload = {"ticker": "INVALID_TICKER_XYZ"}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 404:
        print("   ✅ Correctly returned 404 for invalid ticker")
    else:
        print(f"   ⚠️  Expected 404, got {response.status_code}")
    
    # Test missing ticker
    print("\n2. Testing missing ticker field...")
    payload = {}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 422:
        print("   ✅ Correctly returned 422 for missing field")
    else:
        print(f"   ⚠️  Expected 422, got {response.status_code}")
    
    print("\n✅ Error handling tests complete")


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print("🚀 ATHENA API TEST SUITE")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    print("="*70)
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Supported Stocks", test_stocks),
        ("Single Prediction", lambda: test_predict("AAPL")),
        ("Multiple Predictions", test_multiple_stocks),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except requests.exceptions.ConnectionError:
            print(f"\n❌ CONNECTION ERROR: Cannot connect to {BASE_URL}")
            print("   Make sure the API server is running:")
            print("   python api/main.py")
            break
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()