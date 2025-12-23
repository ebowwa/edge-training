#!/usr/bin/env python3
"""
Smoke test for server bootstrap.
Verifies that the FastAPI app can be created and routes are mounted.
"""

import sys
from pathlib import Path

# Add project root to path
server_dir = Path(__file__).parent.parent
root_dir = server_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(server_dir))

def test_server_bootstrap():
    print("Testing server bootstrap...")
    try:
        from main import app
        print("  ✓ App instantiated successfully")
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/",
            "/api/v1/health",
            "/api/v1/datasets/prepare",
            "/api/v1/train",
            "/api/v1/preprocess"
        ]
        
        for er in expected_routes:
            if any(er in r for r in routes):
                print(f"  ✓ Route found: {er}")
            else:
                print(f"  ✗ Missing route: {er}")
                return False
                
        print("\n✓ Server bootstrap smoke test passed!")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_server_bootstrap()
    sys.exit(0 if success else 1)
