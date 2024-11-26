from ultralytics import YOLO

# Define the model path relative to this script
MODEL_PATH = "prediction/best.pt"

try:
    # Load the model directly using YOLO
    model = YOLO(MODEL_PATH)
    model.model.eval()  # Set the model to evaluation mode
    # Retrieve class names from the model
    class_names = model.names
except Exception as e:
    print(f"Error loading the model: {e}")
    raise e

# Function to perform prediction
def predict(image_path):
    try:
        # Perform inference
        results = model(image_path, save=False, verbose=False)
        top_result = results[0]
        
        # Get the top prediction details
        predicted_index = top_result.probs.top1  # Index of the top-1 class
        confidence = top_result.probs.top1conf  # Confidence of the top-1 class
        predicted_label = class_names[predicted_index]  # Map index to label

        return {"label": predicted_label, "confidence": confidence}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
