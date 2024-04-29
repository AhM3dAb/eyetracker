import logging
import os
import time
from typing import Iterable, NamedTuple, Optional
import math
import cv2
from ultralytics import YOLO
import math
import traceback
import sys
import uvc.uvc_bindings as uvc
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
import numpy as np


model = YOLO('yolov8n.pt')  # load an official model    
names = model.names
# Initialize variables to store the eye center
eye_center = None
pupil_center = None

# Initialize variables for smoothing
history_size = 10  # Adjust the size of the history as needed
pupil_history = []
direction_vector = []
# Initialize variables for Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
kalman.measurementNoiseCov = 1e-2 * np.eye(2, dtype=np.float32)
kalman.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
# Parameters for resetting Kalman filter
max_consecutive_frames_without_detection = 10  # Adjust as needed
consecutive_frames_without_detection = 0
# Initialize variables for last known state
last_known_state = np.array([[0], [0], [0], [0]], dtype=np.float32)


class CameraSpec(NamedTuple):
    name: str
    width: int
    height: int
    fps: int
    bandwidth_factor: float = 2.0

def photoshop_brightness(input_img, brightness = 0):
    ''' input_image:  color or grayscale image
        brightness:  -127 (all black) to +127 (all white)

            returns image of same type as input_image but with
            brightness adjusted

    '''
    img = input_img.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        cv2.convertScaleAbs(input_img, img, alpha_b, gamma_b)

    return img

# def map_point(point, direction, distance_ratio):
#     # Assuming point is a tuple (x, y) in the eye camera plane
#     x, y = point
    
#     # Assuming distance_ratio is the ratio of distances between the two planes (eye, world cameras)
#     mapped_x = x * distance_ratio
#     mapped_y = y * distance_ratio
#     # Map the direction vector
#     mapped_dx = direction[0] * distance_ratio
#     mapped_dy = direction[1] * distance_ratio
#     # Return the new coordinates in the second plane
#     return (mapped_x, mapped_y), (mapped_dx, mapped_dy)

def map_gaze_to_screen(size, direction, eye_center_radius, factor=1):
    gaze = (0,0)
    print(size, direction, eye_center_radius)
    try:
        screen_size = size  # Replace with your actual screen size
        center_x = eye_center[0]
        center_y = eye_center[1]
        intersection_x = center_x + direction[0] * screen_size[0]
        intersection_y = center_y + direction[1] * screen_size[1]
        # Calculate the distance
        world_distance = math.sqrt((intersection_x - center_x)**2 + (intersection_y - center_y)**2)

        distance_ratio = world_distance/eye_center_radius
        pupil_distance = int(math.dist(eye_center,pupil_center)) * (1/factor)

        mapped_distance = pupil_distance*distance_ratio
        mapped_gaze = tuple((np.array(eye_center) + mapped_distance * -direction_vector).astype(int))
        #map_point(pupil_center, direction, distance_ratio)
        print(world_distance)
        print(distance_ratio)
        print(pupil_distance)
        print(mapped_gaze)
        gaze = mapped_gaze
    except Exception as e:
        print(e)
    return gaze


def detect_pupil(data):
    global consecutive_frames_without_detection
    global eye_center
    global last_known_state
    global pupil_center
    global direction_vector
    image_copy = data.copy()
    kalman_state = kalman.predict()
    try:
        pupil_center = None
        
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        ## CLAHE Equalization
        cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cl1.apply(gray)

        ## medianBlur the image to remove noise
        blurred = cv2.medianBlur(clahe, 7)

        # Use HoughCircles to detect circles in the image
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                param1=50, param2=30, minRadius=120, maxRadius=150)
        
        if eye_center is not None and circles is not None:
            circles_test = np.round(circles[0]).astype("int")
            (a, b, r) = circles_test[0]
            if math.dist(eye_center,(a,b)) > 200:     
                circles = None   

        
        if circles is not None:
            # Reset consecutive frames counter
            consecutive_frames_without_detection = 0

            circles = np.round(circles[0]).astype("int")
            (x, y, r) = circles[0]  # Consider the first detected circle
            
            # Draw the circle on the pupil
            #cv2.circle(image_copy, (x, y), r, (0, 255, 0), 4)  # Green circle with thickness 4

            # Draw the center of the circle
            #cv2.circle(image_copy, (x, y), 1, (0, 0, 255), 2)  # Red dot for the center     
           
            # Update the Kalman filter with the measurement
            measurement = np.array([[x], [y]], dtype=np.float32)
            kalman.correct(measurement)
            # Update the last known state
            last_known_state = kalman.statePost.copy()

            # Get the corrected state from the Kalman filter
            kalman_state = kalman.predict()
            
            # Draw the inner circle
            inner_radius = int(r * 0.3)  # Adjust the scale factor as needed
            cv2.circle(image_copy, (int(kalman_state[0]), int(kalman_state[1])), inner_radius, (255, 0, 0), 3)  # Blue circle inside with thickness 4

            # Draw the center of the inner circle
            cv2.circle(image_copy, (int(kalman_state[0]), int(kalman_state[1])), 1, (0, 0, 255), 2)  # Red dot for the center 
            # Set the fixed eye center
            # Set the detected pupil center
            pupil_center = (int(kalman_state[0]), int(kalman_state[1]))
            
            if eye_center is None:
                eye_center = (x,y)
        else:
            #Predict the next state using the Kalman filter
            kalman_state_aux = kalman.predict()
            #No detection in the current frame
            if eye_center is not None:
                if math.dist(eye_center,(int(kalman_state_aux[0]), int(kalman_state_aux[1]))) < 200:     
                    kalman_state = kalman_state_aux

            # Increment consecutive frames counter
            consecutive_frames_without_detection += 1

            # If consecutive frames without detection exceed the threshold, reset Kalman filter
            if consecutive_frames_without_detection > max_consecutive_frames_without_detection:
                kalman.statePost = last_known_state.copy()

            # Draw the outer circle on the pupil using the Kalman state
            # Draw the inner circle
            inner_radius = int(100 * 0.3)  # Adjust the scale factor as needed
            cv2.circle(image_copy, (int(kalman_state[0]), int(kalman_state[1])), inner_radius, (255, 255, 0), 3)  # Blue circle for Kalman state
            pupil_center = (int(kalman_state[0]), int(kalman_state[1]))
            
        # Predict the next state using the Kalman filter
        #kalman_state = kalman.predict()
        # Draw a larger circle to represent the fixed eye center
        if eye_center is not None:
            eye_center_radius = 200  # Adjust the radius as needed
            cv2.circle(image_copy, eye_center, eye_center_radius, (255, 100, 0), 4)  # Blue circle for eye center
        # Draw a line from the eye center to the edge of the pupil
        if eye_center is not None and pupil_center is not None:
            # Calculate the vector from the eye center to the pupil center
            direction_vector = np.array(pupil_center) - np.array(eye_center)
            # Normalize the direction vector
            direction_vector = (direction_vector / np.linalg.norm(direction_vector)).astype(float)
            # Extend the line to the edge of the pupil
            adjusted_line_length = 2*int(math.dist(eye_center,pupil_center))
            endpoint = tuple((np.array(pupil_center) + adjusted_line_length * direction_vector).astype(int))
            cv2.line(image_copy, eye_center, pupil_center, (0, 0, 255), 1)  # Red line representing the direction
            cv2.line(image_copy, pupil_center, endpoint, (0, 255, 0), 2)  # Red line representing the direction
    except Exception as e:
        print(traceback.format_exc())
    return image_copy, pupil_center

def mirror_coordinate_on_screen(coordinates, screen_size, axis='x'):
    screen_width, screen_height = screen_size
    x, y = coordinates

    if axis == 'x':
        mirrored_x = screen_width - x
        mirrored_coordinates = (mirrored_x, y)
    elif axis == 'y':
        mirrored_y = screen_height - y
        mirrored_coordinates = (x, mirrored_y)
    else:
        raise ValueError("Invalid axis. Use 'x' or 'y'.")

    return mirrored_coordinates

def mirror_coordinates_both_axes(coordinates, screen_size):
    screen_width, screen_height = screen_size
    x, y = coordinates

    mirrored_x = screen_width - x
    mirrored_y = screen_height - y

    mirrored_coordinates = (mirrored_x, mirrored_y)
    return mirrored_coordinates
def main(camera_specs: Iterable[CameraSpec]):
    global pupil_data
    devices = uvc.device_list()
    cameras = {spec: init_camera_from_list(devices, spec) for spec in camera_specs}
    if not all(cameras.values()):
        raise RuntimeError(
            "Could not initialize all specified cameras. Available: "
            f"{[dev['name'] for dev in devices]}"
        )

    try:
        keep_running = True
        last_update = time.perf_counter()

        while keep_running:
            for spec, cam in cameras.items():
                try:
                    frame = cam.get_frame(timeout=0.001)
                    
                except TimeoutError:
                    pass
                    # keep_running = False
                    # break
                except uvc.InitError as err:
                    logging.debug(f"Failed to init {spec}: {err}")
                    keep_running = False
                    break
                except uvc.StreamError as err:
                    logging.debug(f"Failed to get a frame for {spec}: {err}")
                else:
                    
                        data = frame.bgr if hasattr(frame, "bgr") else frame.gray
                        if frame.data_fully_received:
                            if (spec.name == 'USB 2.0 Camera'):
                                

                                img, pupil_coordinates = detect_pupil(data) 
                                cv2.imshow(spec.name, img)

                            elif(spec.name == 'USB Camera'):
 
                                results = model(data, stream=True)  # predict on an image                                
                                for r in results:
                                    im_array = r.plot()  # plot a BGR numpy array of predictions
                                gray = cv2.cvtColor(im_array, cv2.COLOR_BGR2GRAY)

                                ## CLAHE Equalization
                                cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                clahe = cl1.apply(gray)

                                ## medianBlur the image to remove noise
                                blurred = cv2.medianBlur(clahe, 7)
                                img = photoshop_brightness(blurred,-100)
                                darkened_frame = cv2.multiply(img, 0.5)
                                #gaze_point = map_gaze_to_screen((spec.width,spec.height), direction_vector, 200)
                                #cv2.circle(img, gaze_point, 20, (255, 0, 0), 3)  # Blue circle for Kalman state
                                #cv2.imshow(spec.name, img)
                                      

            if (time.perf_counter() - last_update) > 1 / 60:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                last_update = time.perf_counter()

    except KeyboardInterrupt:
        pass

    for cam in cameras.values():
        cam.close()


def init_camera_from_list(devices, camera: CameraSpec) -> Optional[uvc.Capture]:
    logging.debug(f"Searching {camera}...")
    logging.debug(devices)
    for device in devices:
        if device["name"] == camera.name:
            logging.debug(f"Found match by name")
            capture = uvc.Capture(device["uid"])
            capture.bandwidth_factor = camera.bandwidth_factor
            for mode in capture.available_modes:
                if mode[:3] == camera[1:4]:  # compare width, height, fps
                    capture.frame_mode = mode
                    return capture
            else:
                logging.warning(
                    f"None of the available modes matched: {capture.available_modes}"
                )
            capture.close()
    else:
        logging.warning(f"No matching camera with name {camera.name!r} found")


if __name__ == "__main__":
    os.environ["LIBUSB_DEBUG"] = "3"
    install_rich_traceback()
    logging.basicConfig(
        level=logging.NOTSET,
        handlers=[RichHandler(level="DEBUG")],
        format="%(message)s",
        datefmt="[%X]",
    )
    # logging.getLogger("uvc").setLevel("INFO")
    main(
        [
            CameraSpec(
                name="USB Camera",
                width=1280,
                height=720,
                fps=30,
                bandwidth_factor=1.6,
            ),
            CameraSpec(
                name="USB 2.0 Camera",
                width=1280,
                height=720,
                fps=30,
                bandwidth_factor=1.6,
            )
         
        ]
    )
