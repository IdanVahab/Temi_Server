# TEMI Cognitive Companion – Real-Time Scenario Recognition System

🧠 A real-time backend system that empowers the TEMI robot to guide cognitively impaired users through everyday tasks by analyzing live video streams and detecting contextual scenarios.

---

## 🔍 Project Overview

This project implements an intelligent backend service for the **TEMI robot**, aimed at assisting users with cognitive difficulties during daily activities in the kitchen.  
The system receives live image frames from TEMI, runs object detection (YOLOv8), motion tracking (Deep SORT), and applies a **Scenario Engine** that determines user intent and safety context.

---

## 🎯 Key Capabilities

- ✅ Real-time object detection (plates, pots, microwave, cutlery, hands, etc.)
- ✅ Temporal reasoning over object presence across multiple frames
- ✅ Motion tracking per object using Deep SORT
- ✅ Scenario-based analysis: pouring food, using cutlery, interacting with microwave
- ✅ Emergency detection (e.g., metal pot placed in microwave)
- ✅ Speech guidance for users based on recognized scenarios

---

## ⚙️ Core Technologies

| Component              | Technology                     |
|------------------------|---------------------------------|
| Backend Framework      | FastAPI (Python)               |
| Object Detection       | YOLOv8                         |
| Object Tracking        | Deep SORT                      |
| Communication          | WebSocket, MQTT (optional)     |
| Scenario Logic         | Custom engine (Python class)   |
| Speech Output (robot)  | TEMI SDK or TTS via API        |

---

## 🧠 Scenario Engine: How It Works

The `ScenarioHandler` class is the brain of the system.  
It combines:

- 🔄 **Label history** tracking (what objects were seen over time)
- 🧍 **Motion detection** per object (via position deltas of tracked IDs)
- ⏱ **Cooldowns** to prevent repeating scenarios too often
- 📦 **Prioritized scenario evaluation**, including:
  - `metal_pot_in_microwave`: safety hazard (always prioritized)
  - `pouring_food`: pot and plate appear simultaneously
  - `plate_removed_from_microwave`: object disappears while door is open
  - `cutlery_used`: detected + in motion
  - `person_interacts`: person detected and moving

If a scenario is detected and cleared by cooldown logic, it is **returned to the client as a JSON** with optional incident ID.

Example output:
```json
{
  "scenario": "pouring_food",
  "timestamp": 1718700000.123,
  "incident_id": null
}
```
🧪 Example Use Case: Heating Food Safely
User opens microwave

System checks for presence of metal pots

User inserts plate → scenario: plate_inserted_into_microwave

Robot speaks: "Close the microwave when you're ready."

If dangerous behavior is detected (e.g., metal pot in microwave) → robot warns the user

🏗 Project Structure
```
src/
├── routes/                  # WebSocket handler
├── services/
│   ├── yolo_service.py      # YOLO object detection logic
│   └── moon_service.py      # Optional MoonDream API interface
├── utils/
│   └── scenario_handler.py  # ScenarioEngine logic
├── model_train/             # Custom YOLO training scripts
main.py                      # FastAPI entrypoint
.gitignore
requirements.txt
README.md

```
🚀 Getting Started
1. Install dependencies
```
pip install -r requirements.txt
```
2. Run the backend server
```
uvicorn main:app --reload
```
The server exposes a WebSocket endpoint (/ws) for real-time frame analysis and scenario response.

📎 Notes
The model file moondream-0_5b-int8.mf is excluded from GitHub due to size limits (>600MB).
Please download it separately if using MoonDream integration.

Training data and YOLO training weights (.pt) are also ignored for cleanliness.

📦 Related Projects
🤖 Android TEMI Client (coming soon)

👤 Author
Developed by Idan Vahab as part of a machine learning and multimedia final project

Guided by real clinical use-cases for cognitive support

🛡 License
This project is currently private and intended for academic and research purposes.


