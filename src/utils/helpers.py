def normalize_class_names(class_names):
    return set(class_names)

def classify_scenario(normalized):
    if "metal pot in a microwave" in normalized:
        return "metal_pot_in_microwave"
    if "pot" in normalized and "Plate" in normalized:
        return "metal_pot_and_plate"
    if "pot" in normalized:
        return "metal_pot"
    if "Plate" in normalized or "Bowl" in normalized:
        return "plastic_plate"
    if "cutlery" in normalized and "person" in normalized:
        return "utensil_with_hand"
    if "open microwave" in normalized:
        return "microwave_door_open"
    if "closed microwave" in normalized:
        return "microwave_door_closed"
    if "open refrigerator" in normalized:
        return "door_open"
    if "closed refrigerator" in normalized:
        return "fridge_detected"
    return list(normalized)[0] if normalized else "no_objects"
