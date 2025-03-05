from bottle import Bottle, request, response, hook
import os
import base64
import io
from PIL import Image
from image_recognition import ImageRecognizer  # Your existing module, modified

# --- Configuration ---
MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"  # Update with your model path

# --- Bottle App Initialization ---
app = Bottle()

# --- CORS Setup (for Bottle) ---
@app.hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

@app.route('/', method=['OPTIONS', 'GET'])
def index():
    return "Image Recognition API is running!"

# --- YOLO Model Loading (using the ImageRecognizer class) ---
recognizer = ImageRecognizer(MODEL_PATH)

# --- API Routes ---
@app.route('/predict', method=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return {}

    data = request.json
    if not data or 'image' not in data:
        response.status = 400
        return {'error': 'No image provided'}

    image_data = data['image']
    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]

    try:
        # Decode the base64 data.
        image_bytes = base64.b64decode(image_data)

        # Use the ImageRecognizer to get the processed image *as bytes*.
        processed_image_bytes = recognizer.predict_and_draw_boxes(image_bytes)

        # Encode the processed image back to base64 for the response.
        encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')

        # Return the base64 encoded image.
        return {'processed_image': f"data:image/jpeg;base64,{encoded_image}"}

    except ValueError as e:
        response.status = 400
        return {'error': str(e)}
    except Exception as e:
        response.status = 500
        return {'error': 'An unexpected error occurred'}


# --- Image Recognizer Modification (image_recognition.py) ---

# import torch
# from PIL import Image, ImageDraw, UnidentifiedImageError
# import io

# class ImageRecognizer:
#     def __init__(self, model_path):
#         self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
#         self.model.conf = 0.25
#         self.model.iou = 0.45
#         self.model.agnostic = False
#         self.model.multi_label = False
#         self.model.max_det = 1000
#         self.model.eval()  # Put the model in evaluation mode

#     @torch.no_grad()  # Disable gradient calculations
#     def predict_and_draw_boxes(self, image_bytes):
#         """Predicts objects, draws bounding boxes, and returns the image as bytes."""
#         try:
#             img = Image.open(io.BytesIO(image_bytes))
#         except UnidentifiedImageError:
#             raise ValueError("Invalid image data")

#         results = self.model(img, size=640)

#         # Draw bounding boxes on the image
#         draw = ImageDraw.Draw(img)
#         for *xyxy, conf, cls in results.xyxy[0]:
#             xmin, ymin, xmax, ymax = map(int, xyxy)
#             label = f'{self.model.names[int(cls)]} {conf:.2f}'
#             draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
#             draw.text((xmin, ymin - 10), label, fill="red")

#         # Convert the image back to bytes
#         img_byte_arr = io.BytesIO()
#         img.save(img_byte_arr, format='JPEG')  # Or PNG, depending on your needs
#         img_byte_arr = img_byte_arr.getvalue()
#         return img_byte_arr


# --- Run the App (using Bottle's built-in server for development) ---
if __name__ == '__main__':
    # Bottle's built-in server is fine for *development*.
    # For production, use a production WSGI server like Gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=5000, debug=False, reloader=False)

# --- Production Deployment (example with Gunicorn) ---
#   gunicorn --workers 4 --bind 0.0.0.0:8000 app:app

###########################################

# from flask import Flask, request, jsonify
# import sqlite3  
# from flask_cors import CORS  # Import the CORS extension
# # Import the ImageRecognizer class from the image_recognition module
# from image_recognition import ImageRecognizer
# import os

# # --- Configuration ---
# DB_FILE = 'object_detection.db'
# MODEL_PATH = "./KINOKO/yoloresult/okashi23/weights/best.pt"  # Update with your model path

# # --- Flask App Initialization ---
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes in your app

# # --- YOLO Model Loading (using the ImageRecognizer class) ---
# recognizer = ImageRecognizer(MODEL_PATH)

# # --- Database Helper Functions ---
# def get_db_connection():
#     conn = sqlite3.connect(DB_FILE)
#     conn.row_factory = sqlite3.Row  # Access columns by name
#     return conn

# def init_db():
#     conn = get_db_connection()
#     cursor = conn.cursor()

#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS object_counts (
#             object_name TEXT PRIMARY KEY,
#             count INTEGER NOT NULL DEFAULT 0
#         )
#     ''')

#     # Check if the records exists.  Use the CORRECT object names.
#     cursor.execute("SELECT COUNT(*) FROM object_counts WHERE object_name = 'takenoko'")
#     if cursor.fetchone()[0] == 0:
#         cursor.execute("INSERT INTO object_counts (object_name) VALUES ('takenoko')")

#     cursor.execute("SELECT COUNT(*) FROM object_counts WHERE object_name = 'kinoko'")
#     if cursor.fetchone()[0] == 0:
#         cursor.execute("INSERT INTO object_counts (object_name) VALUES ('kinoko')")

#     conn.commit()
#     conn.close()

# # --- API Routes ---
# @app.route('/')
# def index():
#      return "Image Recognition API is running!"
     
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.form:
#         return jsonify({'error': 'No image provided'}), 400

#     image_data = request.form['image']
#     if image_data.startswith("data:image"):
#         image_data = image_data.split(",")[1]

#     # como o app sabe o absolute path que a imagem foi salva?
#         #Results saved to /home/ec2-user/my-image-recognition-app/runs/detect/predict3

#     try:
#         detections = recognizer.predict_image(image_data) # Use the ImageRecognizer

#         # conn = get_db_connection()
#         # if conn:
#         #     try:
#         #         cursor = conn.cursor()
#         #         for detection in detections:
#         #             class_name = detection['class_name']
#         #             update_query = "UPDATE object_counts SET count = count + 1 WHERE object_name = ?"
#         #             cursor.execute(update_query, (class_name,))
#         #         conn.commit()
#         #     except sqlite3.Error as e:
#         #         print(f"Error updating database: {e}")
#         #         conn.rollback()
#         #     finally:
#         #         if cursor:  # Check if cursor exists before closing
#         #             cursor.close()
#         #         conn.close()

#         return jsonify(detections)
#     except ValueError as e:
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         return jsonify({'error': 'An unexpected error occurred'}), 500

# @app.route('/stats', methods=['GET'])
# def get_stats():
#     conn = get_db_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute("SELECT object_name, count FROM object_counts")
#             results = cursor.fetchall()
#             stats = {row['object_name']: row['count'] for row in results}
#             return jsonify(stats)
#         except sqlite3.Error as e:
#             print(f"Error fetching stats: {e}")
#             return jsonify({'error': 'Failed to fetch stats'}), 500
#         finally:
#             cursor.close()
#             conn.close()
#     else:
#       return jsonify({'error': 'Failed to connect to database'}), 500


# # --- Run the App ---

# if __name__ == '__main__':
#     init_db()
#     app.run(debug=True, host='0.0.0.0', port=5000)
