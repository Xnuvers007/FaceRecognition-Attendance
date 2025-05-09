import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import datetime
from mtcnn import MTCNN

# Directory for face image data
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define image dimensions - larger for better feature extraction
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Standard input size for many pre-trained models

# Labels and Data
labels = []
faces = []
label_dict = {}

attendance_cache = set()

# Function to capture face images automatically with enhanced quality
def capture_images_automatically(num_images=100, student_name='Nama', student_class='Kelas'):
    """Captures high-quality face images with varied poses for better training"""
    cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Trying alternative camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No camera available.")
            return
            
    # Set camera resolution to higher quality if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured_count = 0
    
    # Use MTCNN for superior face detection
    detector = MTCNN()
    
    # Create directory for this student if it doesn't exist
    student_dir = os.path.join(data_dir, f"{student_name}_{student_class}")
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    
    print("Starting image capture. Please move your face slightly between captures for variety.")
    print("Look straight, then slightly up, down, left, right for better training data.")
    
    while captured_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces_detected = detector.detect_faces(frame)
        
        for face in faces_detected:
            confidence = face['confidence']
            # Only process high-confidence detections
            if confidence < 0.95:
                continue
                
            bounding_box = face['box']
            x, y, w, h = bounding_box
            
            # Add margin to capture full face (20% extra on each side)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Ensure coordinates are within frame
            x_extended = max(0, x - margin_x)
            y_extended = max(0, y - margin_y)
            w_extended = min(frame.shape[1] - x_extended, w + 2 * margin_x)
            h_extended = min(frame.shape[0] - y_extended, h + 2 * margin_y)
            
            # Extract face with margin
            face_img = frame[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended]
            
            # Skip if face extraction failed
            if face_img.size == 0:
                continue
                
            # Resize the face image to standard size
            face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
            
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display progress text
            cv2.putText(frame, f"Capturing: {captured_count+1}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save the face image
            filename = f"{student_name}_{student_class}_{captured_count+1}.jpg"
            file_path = os.path.join(student_dir, filename)
            cv2.imwrite(file_path, face_img)
            
            captured_count += 1
            print(f"Image {captured_count}/{num_images} saved")
            
            # Short delay to allow slight movement between captures
            cv2.waitKey(300)
        
        # Show the frame with face detection
        cv2.imshow('Face Capture', frame)
        
        # Press ESC to exit early
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total {captured_count} images captured for {student_name}.")
    
    # If we didn't capture enough images, warn the user
    if captured_count < num_images:
        print(f"Warning: Only captured {captured_count} of {num_images} requested images.")

# Function to load and preprocess images for training
def load_dataset():
    """Load and preprocess images from the data directory structure"""
    faces = []
    labels = []
    label_dict = {}
    label_idx = 0
    
    # Iterate through data directory (each folder is a person)
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        
        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue
            
        # Extract name from folder
        person_name = person_folder.split('_')[0]
        
        # Add to label dictionary if new
        if person_name not in label_dict:
            label_dict[person_name] = label_idx
            label_idx += 1
        
        # Process each image in the person's folder
        person_images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not person_images:
            print(f"Warning: No images found for {person_name}")
            continue
            
        print(f"Loading {len(person_images)} images for {person_name}")
        
        for img_file in person_images:
            img_path = os.path.join(person_path, img_file)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # Normalize pixel values
            img = img / 255.0
            
            faces.append(img)
            labels.append(label_dict[person_name])
    
    if not faces:
        print("Error: No valid images found in the dataset!")
        return None, None, None
        
    # Convert to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    
    print(f"Dataset loaded: {len(faces)} images, {len(label_dict)} unique individuals")
    
    return faces, labels, label_dict

# Function to create a network that learns face embeddings
def create_embedding_model():
    """Create a model that extracts facial embeddings for better unknown face detection"""
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),  # This will be our embedding vector
    ])
    
    return model

# Function to train a high-accuracy model with transfer learning and embeddings
def train_model():
    """Train a high-accuracy face recognition model using transfer learning and face embeddings"""
    # Load and preprocess the dataset
    faces, labels, label_dict = load_dataset()
    
    if faces is None or len(faces) == 0:
        print("Error: No training data available. Please capture images first.")
        return
    
    # Save label dictionary for prediction
    np.save('label_dict.npy', label_dict)
    
    # Convert labels to categorical format
    num_classes = len(set(labels))
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    # Split data into training and validation sets (80-20 split)
    indices = np.arange(faces.shape[0])
    np.random.shuffle(indices)
    faces = faces[indices]
    labels_categorical = labels_categorical[indices]
    
    split_idx = int(len(faces) * 0.8)
    X_train, X_val = faces[:split_idx], faces[split_idx:]
    y_train, y_val = labels_categorical[:split_idx], labels_categorical[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    
    # Data augmentation for training
    data_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # First, create and train the embedding model
    embedding_model = create_embedding_model()
    
    # Create the classification model
    model = tf.keras.Sequential([
        embedding_model,
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        data_augmentation.flow(X_train, y_train, batch_size=16),
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning: Unfreeze some layers of the base model
    print("Fine-tuning the model...")
    
    # Unfreeze the top layers of the base model
    for layer in embedding_model.layers[0].layers[-30:]:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history_fine = model.fit(
        data_augmentation.flow(X_train, y_train, batch_size=16),
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    eval_result = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {eval_result[1]*100:.2f}%")
    
    # Save the final model
    model.save('face_recognition_model.h5')
    
    # Save embedding model separately for calculating face similarity
    embedding_model.save('face_embedding_model.h5')
    
    # Extract and save embeddings for all known faces to improve unknown detection
    print("Calculating embeddings for known faces...")
    all_embeddings = []
    all_names = []
    
    # For each person in the dataset, calculate average embeddings
    for person_name, person_idx in label_dict.items():
        person_label_indices = np.where(labels == person_idx)[0]
        person_faces = faces[person_label_indices]
        
        # Calculate embeddings for all faces of this person
        person_embeddings = embedding_model.predict(person_faces, verbose=0)
        
        # Calculate average embedding for this person
        avg_embedding = np.mean(person_embeddings, axis=0)
        all_embeddings.append(avg_embedding)
        all_names.append(person_name)
    
    # Save the embeddings and corresponding names
    np.save('known_face_embeddings.npy', np.array(all_embeddings))
    np.save('known_face_names.npy', np.array(all_names))
    
    print("Model trained and saved as 'face_recognition_model.h5'")
    print("Embedding model saved as 'face_embedding_model.h5'")
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Combine histories
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved as 'training_history.png'")
    except:
        print("Couldn't generate training history plot. Matplotlib may not be installed.")

# Calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to predict faces with high accuracy and improved unknown detection
def predict_face_from_live_camera():
    """Recognize faces from live camera with high accuracy and improved unknown detection"""
    
    # Check if models exist
    if not os.path.exists('face_recognition_model.h5'):
        print("Error: Model not found. Please train the model first.")
        return
        
    if not os.path.exists('face_embedding_model.h5'):
        print("Error: Embedding model not found. Please train the model first.")
        return
        
    # Check if embeddings exist
    if not os.path.exists('known_face_embeddings.npy') or not os.path.exists('known_face_names.npy'):
        print("Error: Known face embeddings not found. Please train the model first.")
        return
    
    # Load models and data
    model = load_model('face_recognition_model.h5')
    embedding_model = load_model('face_embedding_model.h5')
    label_dict = np.load('label_dict.npy', allow_pickle=True).item()
    known_embeddings = np.load('known_face_embeddings.npy')
    known_names = np.load('known_face_names.npy')
    
    # Reverse the label dictionary for prediction
    idx_to_label = {idx: name for name, idx in label_dict.items()}
    
    # Initialize camera
    cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Trying alternative camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No camera available.")
            return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    print("Face recognition started. Press ESC to exit.")
    
    # Attendance log to avoid duplicate entries (name: last_logged_time)
    attendance_log_cache = {}
    
    # Time interval between logs for the same person (in seconds)
    log_interval = 300  # 5 minutes
    
    # Similarity threshold for unknown detection
    similarity_threshold = 0.70  # Adjust as needed based on your dataset
    
    # Softmax confidence threshold
    confidence_threshold = 0.70  # Initial threshold, can be adjusted
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            break
        
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Detect faces
        faces_detected = detector.detect_faces(frame)
        
        for face in faces_detected:
            confidence = face['confidence']
            
            # Only process high-confidence detections
            if confidence < 0.9:
                continue
                
            bounding_box = face['box']
            x, y, w, h = bounding_box
            
            # Add margin to capture full face
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Ensure coordinates are within frame
            x_extended = max(0, x - margin_x)
            y_extended = max(0, y - margin_y)
            w_extended = min(frame.shape[1] - x_extended, w + 2 * margin_x)
            h_extended = min(frame.shape[0] - y_extended, h + 2 * margin_y)
            
            # Extract face
            face_img = frame[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended]
            
            if face_img.size == 0:
                continue
                
            # Preprocess for prediction
            face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_normalized = face_img_rgb / 255.0
            face_input = np.expand_dims(face_img_normalized, axis=0)
            
            # Get face embedding
            face_embedding = embedding_model.predict(face_input, verbose=0)[0]
            
            # Calculate similarity to known faces
            max_similarity = -1
            most_similar_name = None
            
            for i, known_embedding in enumerate(known_embeddings):
                similarity = cosine_similarity(face_embedding, known_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_name = known_names[i]
            
            # Standard prediction from classifier
            prediction = model.predict(face_input, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            prediction_confidence = prediction[0][predicted_class_idx]
            classifier_name = idx_to_label.get(predicted_class_idx, "Unknown")
            
            # Decision logic combining embedding similarity and classifier confidence
            if max_similarity > similarity_threshold and prediction_confidence > confidence_threshold:
                # Check if the two methods agree
                if most_similar_name == classifier_name:
                    # High confidence match
                    predicted_name = most_similar_name
                    confidence_value = (max_similarity + prediction_confidence) / 2  # Average the confidences
                    box_color = (0, 255, 0)  # Green
                else:
                    # Methods disagree, use the more confident one
                    if max_similarity > prediction_confidence:
                        predicted_name = most_similar_name
                        confidence_value = max_similarity
                    else:
                        predicted_name = classifier_name
                        confidence_value = prediction_confidence
                    box_color = (0, 255, 255)  # Yellow
            else:
                # Low confidence - label as unknown
                predicted_name = "Unknown"
                confidence_value = max(max_similarity, prediction_confidence)
                box_color = (0, 0, 255)  # Red

            if confidence_value < 0.9 and predicted_name != "Unknown":
                predicted_name = "ragu (Doubt)/Unknown"
                box_color = (0, 255, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Display info
            result_text = f"{predicted_name}: {confidence_value*100:.1f}%"
            cv2.putText(display_frame, result_text, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            if predicted_name != "Unknown" and "ragu" not in predicted_name.lower():
                if predicted_name not in attendance_cache:
                    log_attendance(predicted_name)
                    attendance_cache.add(predicted_name)  # Tambahkan ke cache

            # Log attendance only for known faces
            if predicted_name != "Unknown":
                current_time = datetime.datetime.now()
                last_logged_time = attendance_log_cache.get(predicted_name)
                
                if (last_logged_time is None or 
                    (current_time - last_logged_time).total_seconds() > log_interval):
                    log_attendance(predicted_name)
                    attendance_log_cache[predicted_name] = current_time
        
        # Display frame
        cv2.imshow('Face Recognition', display_frame)
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to log attendance
def log_attendance(name):
    """Log attendance with timestamp"""
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    date = current_time.strftime('%Y-%m-%d')
    
    # Log to daily file
    daily_log_file = f'attendance_{date}.csv'
    
    # Check if file exists, if not create with header
    file_exists = os.path.isfile(daily_log_file)
    
    with open(daily_log_file, 'a') as f:
        if not file_exists:
            f.write('Timestamp,Name\n')
        f.write(f'{timestamp},{name}\n')
    
    # Also log to master file
    master_file = 'attendance_master_log.csv'
    master_file_exists = os.path.isfile(master_file)
    
    with open(master_file, 'a') as f:
        if not master_file_exists:
            f.write('Date,Time,Name\n')
        f.write(f'{date},{current_time.strftime("%H:%M:%S")},{name}\n')
    
    print(f'Attendance logged: {name} at {timestamp}')

# Function to view attendance logs
def view_attendance_logs():
    """View and analyze attendance logs"""
    print("\n=== Attendance Log Viewer ===")
    
    # Check if any logs exist
    log_files = [f for f in os.listdir('.') if f.startswith('attendance_') and f.endswith('.csv')]
    
    if not log_files:
        print("No attendance logs found.")
        return
    
    print("Available attendance logs:")
    for i, file in enumerate(sorted(log_files), 1):
        print(f"{i}. {file}")
    
    choice = input("\nEnter the number of the log to view, or 'a' for all: ")
    
    if choice.lower() == 'a':
        # Display summary of all logs
        attendance_summary = {}
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    
                    name = parts[1]
                    if name not in attendance_summary:
                        attendance_summary[name] = 0
                    attendance_summary[name] += 1
        
        print("\n=== Attendance Summary ===")
        print("Name\t\tTotal Appearances")
        print("-" * 30)
        
        for name, count in sorted(attendance_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}\t\t{count}")
    
    elif choice.isdigit() and 1 <= int(choice) <= len(log_files):
        selected_file = sorted(log_files)[int(choice) - 1]
        
        print(f"\nViewing log: {selected_file}")
        print("-" * 40)
        
        with open(selected_file, 'r') as f:
            lines = f.readlines()
            
            # Print header
            print(lines[0].strip())
            print("-" * 40)
            
            # Print entries
            for line in lines[1:]:
                print(line.strip())
    
    else:
        print("Invalid choice.")

# Main menu
def main_menu():
    """Main program menu"""
    while True:
        print("\n" + "="*50)
        print("   IMPROVED FACE RECOGNITION ATTENDANCE SYSTEM   ")
        print("="*50)
        print("1. Capture student face images")
        print("2. Train recognition model (95%+ accuracy)")
        print("3. Start face recognition attendance")
        print("4. View attendance logs")
        print("5. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            student_name = input("Enter student name: ")
            student_class = input("Enter student class: ")
            num_images = input("Number of images to capture (recommended: 100): ")
            
            try:
                num_images = int(num_images)
            except ValueError:
                num_images = 100
                
            capture_images_automatically(num_images=num_images, 
                                       student_name=student_name, 
                                       student_class=student_class)
        
        elif choice == '2':
            print("\nTraining high-accuracy model with improved unknown detection...")
            print("This may take several minutes depending on your dataset size and hardware.")
            train_model()
        
        elif choice == '3':
            print("\nStarting improved face recognition...")
            predict_face_from_live_camera()
        
        elif choice == '4':
            view_attendance_logs()
        
        elif choice == '5':
            print("\nThank you for using the Face Recognition Attendance System!")
            break
        
        else:
            print("Invalid choice. Please try again.")

# Run the program
if __name__ == "__main__":
    print("Starting Improved Face Recognition Attendance System...")
    main_menu()