from collections import deque
import time
import math

class ScenarioHandler:
    def __init__(self, max_history=10):
        self.label_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        self.last_pour_time = 0
        self.POUR_COOLDOWN = 4  # seconds

        # Tracking תנועה לפי Track ID
        self.track_history = {}  # track_id -> deque of bbox positions
        self.label_by_track_id = {}  # track_id -> label
        self.MOVEMENT_THRESHOLD = 15  # pixels (euclidean distance)
        self.MAX_TRACK_HISTORY = 5

        # ניהול תרחישים אחרונים ו־cooldown
        self.last_reported_scenario = None
        self.last_report_time = 0
        self.scenario_cooldowns = {
            "pouring_food": 5,
            "plate_inserted_into_microwave": 5,
            "plate_removed_from_microwave": 5,
            "pot_and_plate_on_counter": 5,
            "cutlery_detected": 5,
            "plate_moved": 5,
            "pot_moved": 5,
            "cutlery_used": 5,
            "person_interacts": 5,
            "metal_pot_in_microwave": 1,  # חירום – דיווח מהיר
        }
        self.incident_counters = {}

    def update(self, labels: set):
        now = time.time()
        self.label_history.append(labels)
        self.timestamp_history.append(now)

    def update_tracking(self, tracked_objects: list):
        for obj in tracked_objects:
            track_id = obj["id"]
            bbox = obj["bbox"]  # [left, top, width, height]
            label = obj["label"]
            self.label_by_track_id[track_id] = label
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            point = (center_x, center_y)

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.MAX_TRACK_HISTORY)
            self.track_history[track_id].append(point)

    def is_moving(self, track_id: int) -> bool:
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return False
        p1 = self.track_history[track_id][-2]
        p2 = self.track_history[track_id][-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance > self.MOVEMENT_THRESHOLD

    def detect_pouring_scenario(self) -> bool:
        now = time.time()
        pot_times = []
        plate_times = []

        for labels, ts in zip(self.label_history, self.timestamp_history):
            if "pot" in labels:
                pot_times.append(ts)
            if "Plate" in labels or "Bowl" in labels:
                plate_times.append(ts)

        if pot_times and plate_times:
            if abs(pot_times[-1] - plate_times[-1]) < 1.5:
                if now - self.last_pour_time > self.POUR_COOLDOWN:
                    self.last_pour_time = now
                    return True
        return False

    def detect_plate_removed_scenario(self) -> bool:
        if len(self.label_history) < 2:
            return False

        prev_labels = self.label_history[-2]
        curr_labels = self.label_history[-1]

        prev_has_plate = "Plate" in prev_labels or "Bowl" in prev_labels
        curr_has_plate = "Plate" in curr_labels or "Bowl" in curr_labels

        microwave_open = "open microwave" in curr_labels or "open microwave" in prev_labels

        if prev_has_plate and not curr_has_plate and microwave_open:
            return True
        return False

    def detect_pot_and_plate_on_counter(self) -> bool:
        if not self.label_history:
            return False
        labels = self.label_history[-1]
        return "pot" in labels and "Plate" in labels

    def detect_cutlery_present(self) -> bool:
        return any("cutlery" in labels for labels in self.label_history)

    def detect_plate_inserted_into_microwave(self) -> bool:
        if not self.label_history:
            return False
        labels = self.label_history[-1]
        return "Plate" in labels and "open microwave" in labels

    def detect_plate_moved(self) -> bool:
        for track_id, label in self.label_by_track_id.items():
            if label == "Plate" and self.is_moving(track_id):
                return True
        return False

    def detect_pot_moved(self) -> bool:
        for track_id, label in self.label_by_track_id.items():
            if label == "pot" and self.is_moving(track_id):
                return True
        return False

    def detect_cutlery_used(self) -> bool:
        for track_id, label in self.label_by_track_id.items():
            if label == "cutlery" and self.is_moving(track_id):
                return True
        return False

    def detect_person_interacts(self) -> bool:
        for track_id, label in self.label_by_track_id.items():
            if label == "person" and self.is_moving(track_id):
                return True
        return False

    def detect_metal_pot_in_microwave(self) -> bool:
        if not self.label_history:
            return False
        labels = self.label_history[-1]
        return "metal_pot_in_microwave" in labels

    def should_send_scenario(self, scenario_name):
        now = time.time()
        cooldown = self.scenario_cooldowns.get(scenario_name, 5)

        if scenario_name == "metal_pot_in_microwave":
            count = self.incident_counters.get(scenario_name, 0) + 1
            self.incident_counters[scenario_name] = count
            incident_id = f"{scenario_name}_incident_{count}"
            return True, incident_id

        if self.last_reported_scenario != scenario_name or (now - self.last_report_time > cooldown):
            self.last_reported_scenario = scenario_name
            self.last_report_time = now
            return True, None

        return False, None

    def get_active_scenario(self):
        scenario_order = [
            ("metal_pot_in_microwave", self.detect_metal_pot_in_microwave),
            ("pouring_food", self.detect_pouring_scenario),
            ("plate_removed_from_microwave", self.detect_plate_removed_scenario),
            ("pot_and_plate_on_counter", self.detect_pot_and_plate_on_counter),
            ("cutlery_detected", self.detect_cutlery_present),
            ("plate_inserted_into_microwave", self.detect_plate_inserted_into_microwave),
            ("plate_moved", self.detect_plate_moved),
            ("pot_moved", self.detect_pot_moved),
            ("cutlery_used", self.detect_cutlery_used),
            ("person_interacts", self.detect_person_interacts),
        ]

        for scenario_name, detector in scenario_order:
            if detector():
                should_send, incident_id = self.should_send_scenario(scenario_name)
                if should_send:
                    print(f"🧠 Sending scenario: {scenario_name}, incident: {incident_id}")
                    return {
                        "scenario": scenario_name,
                        "timestamp": time.time(),
                        "incident_id": incident_id
                    }
                else:
                    print(f"⚠ Skipping duplicate scenario: {scenario_name}")
                    return None

        self.last_reported_scenario = None
        return None
