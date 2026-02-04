# Automotive-Preventive-Pedastrian-Protection-function-code

from ultralytics import YOLO
import cv2


#Load the exported ONNX model

onnx_model = YOLO("yolo11n.onnx")

#Classes to keep

allowed_classes = {
    "person", "bicycle", "motorcycle", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
}

#Initialize camera

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference
    
    results = onnx_model.predict(source=frame, show=False, stream=False)

    # Get detection result for the first frame
    
    result = results[0]

    # Copy frame for annotation
    
    filtered_frame = frame.copy()

    # Filter and draw boxes
    
    for box in result.boxes:
        cls_id = int(box.cls)
        
        # Check if class_id is within bounds
        
        if cls_id >= len(result.names):
            print(f"Warning: class_id {cls_id} is out of range!")
            continue
        
        cls_name = result.names[cls_id]

        if cls_name in allowed_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"{cls_name} {conf:.2f}"

            # Draw box and label
            
            cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(filtered_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Display the annotated frame
    
    cv2.imshow("Filtered YOLOv11n ONNX Stream", filtered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
