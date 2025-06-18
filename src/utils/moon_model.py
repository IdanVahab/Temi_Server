import moondream as md
import cv2
from PIL import Image

md_model = md.vl(model="C:/Users/Idan Vahab/Desktop/TemiSafetyApp/src/moondream-0_5b-int8.mf")

def describe_image_with_moondream(image_np, labels):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).resize((320, 240))
        encoded = md_model.encode_image(image_pil)
        question = get_moondream_question(labels)
        print(f"❓ MoonDream question: {question}")
        result = md_model.query(encoded, question)
        return result.get("answer", "No description")
    except Exception as e:
        print(f"❌ MoonDream error: {e}")
        return "MoonDream failed to generate description."

def get_moondream_question(labels: set) -> str:
    if "metal pot in a microwave" in labels:
        return "Is there a metal pot inside the microwave?"
    if "pot" in labels and "Plate"  in labels:
        return "Is someone pouring food from a pot to a plate?"
    if "pot" in labels and "Bowl"  in labels:
        return "Is someone pouring food from a pot to a Bowl?"

    return "Does the image show a kitchen-related action? yes or no?" 
