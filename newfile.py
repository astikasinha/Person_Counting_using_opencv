import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0) 

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
mid_line = frame_height // 3
second_line = mid_line + 1  # 1 pixel gap below the red line

# Initialize trackers for detected faces
trackers = {}
next_id = 1
crossing_count_first_line = 0  # For the first line
crossing_count_second_line = 0  # For the second line

# Helper function to determine which side of the line a y-coordinate is on
def get_line_side(y, line_position):
    return "above" if y < line_position else "below"

while cap.isOpened():
    for _ in range(4):  # Adjust the number as needed
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw the first red line (mid line)
    cv2.line(frame, (0, mid_line), (frame_width, mid_line), (0, 0, 255), 2)

    # Draw the second line 1 pixel below the red line
    cv2.line(frame, (0, second_line), (frame_width, second_line), (255, 0, 0), 2)

    updated_trackers = {}

    # Process each detected face
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        # Find if the face matches an existing tracker
        matched_id = None
        for track_id, (prev_x, prev_y, prev_side_first_line, prev_side_second_line) in trackers.items():
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
            if distance < 50:  
                matched_id = track_id
                break

        if matched_id is None:
            # Assign a new ID if no match is found
            matched_id = next_id
            next_id += 1

        # Update the tracker
        current_side_first_line = get_line_side(center_y, mid_line)
        current_side_second_line = get_line_side(center_y, second_line)

        # Check if the face crosses the first line (mid line)
        if trackers.get(matched_id, (0, 0, "none", "none"))[2] != current_side_first_line:
            if trackers.get(matched_id, (0, 0, "none", "none"))[2] == "above" and current_side_first_line == "below":
                crossing_count_first_line += 1
            elif trackers.get(matched_id, (0, 0, "none", "none"))[2] == "below" and current_side_first_line == "above":
                crossing_count_first_line += 1

        # Check if the face crosses the second line (1 pixel below the red line)
        if trackers.get(matched_id, (0, 0, "none", "none"))[3] != current_side_second_line:
            if trackers.get(matched_id, (0, 0, "none", "none"))[3] == "above" and current_side_second_line == "below":
                crossing_count_second_line += 1
            elif trackers.get(matched_id, (0, 0, "none", "none"))[3] == "below" and current_side_second_line == "above":
                crossing_count_second_line += 1

        updated_trackers[matched_id] = (center_x, center_y, current_side_first_line, current_side_second_line)

        # Draw rectangle and ID on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Update trackers for the next frame
    trackers = updated_trackers

    # Determine the maximum of both counts as the final count
    final_count = max(crossing_count_first_line, crossing_count_second_line)

    # Display the maximum crossing count
    cv2.putText(
        frame, f"Final Count: {final_count}", (10, 30),
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
