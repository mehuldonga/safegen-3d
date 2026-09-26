"""
API Tests for SAFEGEN 3D Backend Endpoints.
Tests /health, /environment, /planning/create, /planning/validate, /metrics.
Classification: [ENGINEERING ADDITION]
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from fastapi.testclient import TestClient
    from app.backend.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestAPIEndpoints(unittest.TestCase):
    """Test REST endpoints with valid and invalid requests."""

    def setUp(self):
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI is not yet available in the environment")
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test GET /health returns 200 and expected status fields."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("simulation_status", data)

    def test_environment_endpoint(self):
        """Test GET /environment returns maze layout dimensions."""
        response = self.client.get("/environment")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("layout", data)
        self.assertIn("width", data)
        self.assertIn("height", data)

    def test_mission_creation_valid(self):
        """Test POST /planning/create with valid coordinates."""
        payload = {
            "start": [1.0, 1.0],
            "goal": [7.0, 7.0]
        }
        response = self.client.post("/planning/create", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["start"], [1.0, 1.0])

    def test_mission_creation_invalid(self):
        """Test POST /planning/create with missing fields returns 422 Unprocessable Entity."""
        payload = {"start": [1.0, 1.0]}  # Missing goal
        response = self.client.post("/planning/create", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_obstacle_injection(self):
        """Test POST /simulation/obstacle adds obstacle to digital twin."""
        payload = {
            "position": [4.0, 4.0],
            "radius": 0.9,
            "shape": "ellipse",
            "label": "Dynamic Hazard"
        }
        response = self.client.post("/simulation/obstacle", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_metrics_endpoint(self):
        """Test GET /metrics returns benchmark metrics object."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("planning_time_ms", data)
        self.assertIn("safety_violations", data)
        self.assertIn("min_safety_margin", data)


if __name__ == "__main__":
    unittest.main()
