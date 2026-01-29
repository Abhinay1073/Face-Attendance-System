import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace


DB_PATH = "face_dbb"
LOG_FILE = "attendance.csv"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)


def register_user():

    name = input("Enter name to register: ").strip()
    cap = cv2.VideoCapture(0)

    print(f"Registering {name}. CLICK THE CAMERA WINDOW, then press 'S' to save.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.putText(frame, f"Registering: {name} | Press 'S' to Save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Registration Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            img_path = os.path.join(DB_PATH, f"{name}.jpg")
            cv2.imwrite(img_path, frame)


            cache_path = os.path.join(DB_PATH, "representations_vgg_face.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)

            print(f"SUCCESS: {name} saved to {DB_PATH}.")
            break
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_attendance():

    cap = cv2.VideoCapture(0)
    print("\n--- Attendance System Active ---")
    print("Press 'P' for Punch-In | 'O' for Punch-Out | 'Q' to Exit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Attendance Feed - Press P or O", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("Exiting project...")
            break

        if key in [ord('p'), ord('P'), ord('o'), ord('O')]:
            action = "Punch-In" if key in [ord('p'), ord('P')] else "Punch-Out"
            print(f"Processing {action}...")

            try:

                results = DeepFace.find(img_path=frame,
                                        db_path=DB_PATH,
                                        model_name="VGG-Face",
                                        enforce_detection=False,  # Change to False
                                        anti_spoofing=False)  # Change to False for testing

                if len(results) > 0 and not results[0].empty:
                    user_path = results[0].iloc[0]['identity']
                    user_name = os.path.basename(user_path).split('.')[0]


                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.DataFrame([{"Name": user_name, "Time": now, "Action": action}])
                    df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
                    print(f"VERIFIED: {user_name} - {action} logged.")
                else:
                    print("REJECTED: Face not recognized.")
            except Exception as e:
                print("AUTH FAILED: Liveness check failed or no face detected.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("Select: (1) Register User (2) Start Attendance (Q) Quit: ").strip().lower()
    if choice == '1':
        register_user()
    elif choice == '2':
        start_attendance()
    else:
        exit()

