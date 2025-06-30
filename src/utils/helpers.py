def normalize_class_names(class_names):
    """
    Converts a list of class names (as returned by YOLO) into a set of unique labels.

    Args:
        class_names (list): List of class name strings (possibly with duplicates).

    Returns:
        set: A set containing unique class names.
    """
    return set(class_names)


def classify_scenario(normalized):
    """
    Classifies the current scenario based on the set of detected object labels.

    The classification is done using a priority-based rule set, where specific combinations
    of objects map to specific high-level scenario labels.

    Args:
        normalized (set): A set of object labels detected in the current frame.

    Returns:
        str: Scenario name (e.g., "metal_pot_in_microwave", "plastic_plate").
             If no known pattern is matched, returns the first label found or "no_objects".
    """
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

    # Fallback: return any detected label or indicate that no objects were found
    return list(normalized)[0] if normalized else "no_objects"
