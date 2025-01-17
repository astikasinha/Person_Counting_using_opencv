import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0) 

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
mid_line = frame_height // 3 

# Initialize trackers for detected faces
trackers = {}
next_id = 1
crossing_count = 0

# Helper function to determine which side of the line a y-coordinate is on
def get_line_side(y, mid_line):
    return "above" if y < mid_line else "below"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw the red line in the middle
    cv2.line(frame, (0, mid_line), (frame_width, mid_line), (0, 0, 255), 2)

    updated_trackers = {}

    # Process each detected face
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        # Find if the face matches an existing tracker
        matched_id = None
        for track_id, (prev_x, prev_y, prev_side) in trackers.items():
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
            if distance < 50:  
                matched_id = track_id
                break

        if matched_id is None:
            # Assign a new ID if no match is found
            matched_id = next_id
            next_id += 1

        # Update the tracker
        current_side = get_line_side(center_y, mid_line)
        if trackers.get(matched_id, (0, 0, "none"))[2] != current_side:
            if trackers.get(matched_id, (0, 0, "none"))[2] == "above" and current_side == "below":
                crossing_count += 1
            elif trackers.get(matched_id, (0, 0, "none"))[2] == "below" and current_side == "above":
                crossing_count += 1

        updated_trackers[matched_id] = (center_x, center_y, current_side)

        # Draw rectangle and ID on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Update trackers for the next frame
    trackers = updated_trackers

    # Display the crossing count
    cv2.putText(
        frame, f"Count: {crossing_count}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the frame with the detected faces
    cv2.imshow('Webcam Face Detection', frame)

    # Press 'q' to exit the webcam loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
