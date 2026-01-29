import requests
import time

# Configuration
API_URL = "http://127.0.0.1:8000/predict"

def test_scenario(name, data):
    print(f"\n--- Testing: {name} ---")
    try:
        start = time.time()
        response = requests.post(API_URL, json=data)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success ({latency:.1f}ms)")
            print(f"   Status: {result.get('status')}")
            print(f"   Risk:   {result.get('risk_score'):.1%}")
            print(f"   Msg:    {result.get('message')}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Failed. Is the server running? ({e})")

if __name__ == "__main__":
    # Test 1: Normal Operation (Quench)
    test_scenario("Normal Operation (Quench)", {
        "pressure": 3.5,
        "drift": 0.0,
        "r2": 0.99,
        "temp": 900,
        "scan_speed": 10,
        "machine_state": "QUENCH"
    })

    # Test 2: Heating (Ignored)
    test_scenario("Heating Phase (Should be Ignored)", {
        "pressure": 0.0,
        "drift": 0.0,
        "r2": 0.0,
        "temp": 200,
        "scan_speed": 0,
        "machine_state": "HEATING"
    })
    
    # Test 3: Critical Failure
    test_scenario("Simulated Failure", {
        "pressure": 3.2,
        "drift": -0.06,
        "r2": 0.99,
        "temp": 900,
        "scan_speed": 10,
        "machine_state": "QUENCH"
    })
