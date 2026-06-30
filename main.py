import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from collections import Counter
import csv
import difflib
import math
import os
import time
import cv2
from engine import VehicleEngine

VEHICLE_CLASSES = {
    'car': 'car',
    'motorcycle': 'bike',
    'bicycle': 'bike',
    'truck': 'truck',
    'bus': 'bus'
}

def is_duplicate_plate(plate, logged_plates):
    for lp in logged_plates:
        if lp in plate or plate in lp:
            if min(len(lp), len(plate)) >= 4:
                return True
        if difflib.SequenceMatcher(None, plate, lp).ratio() > 0.70:
            return True
    return False

def run_pipeline(video_path=None):
    engine = VehicleEngine()
    
    if not video_path or not os.path.exists(video_path):
        if os.path.exists('Traffic Control CCTV.mp4'):
            video_path = 'Traffic Control CCTV.mp4'
        else:
            sample_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
            if sample_files:
                video_path = sample_files[0]
            else:
                print("Error: No valid video file found.")
                return
    print(f"Using video file: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return

    csv_file = 'detection.csv'
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    logged_plates = set()
    if file_exists:
        with open(csv_file, 'r', encoding='utf-8') as rf:
            reader = csv.reader(rf)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[3] != "Unknown":
                    logged_plates.add(row[3])

    trackers = []
    frame_count = 0
    cached_vehicles = []
    cached_plates = []

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'Vehicle', 'Color', 'Plate'])
            f.flush()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Execute AI inference every 2nd frame to ensure smooth real-time CPU performance
            if frame_count % 2 == 1:
                v_res, p_res = engine.detect(frame)
                cached_vehicles = []
                for r in v_res:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]
                        if label in VEHICLE_CLASSES:
                            v_type = VEHICLE_CLASSES[label]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = frame[y1:y2, x1:x2]
                            color = engine.get_color(crop)
                            cached_vehicles.append({
                                'box': (x1, y1, x2, y2),
                                'type': v_type,
                                'color': color
                            })

                unique_candidates = []
                for r in p_res:
                    for box in r.boxes:
                        gx1, gy1, gx2, gy2 = map(int, box.xyxy[0])
                        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                        if not any(math.hypot(gcx - ((u[0] + u[2]) / 2), gcy - ((u[1] + u[3]) / 2)) < 40 for u in unique_candidates):
                            unique_candidates.append((gx1, gy1, gx2, gy2))
                cached_plates = unique_candidates

            for veh in cached_vehicles:
                vx1, vy1, vx2, vy2 = veh['box']
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(frame, f"{veh['type'].upper()} | {veh['color']}", (vx1, max(vy1 - 10, 15)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            for pbox in cached_plates:
                x1, y1, x2, y2 = pbox
                if (x2 - x1) < 15 or (y2 - y1) < 8:
                    continue

                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                matched_tracker = None
                best_t_dist = float('inf')
                for t in trackers:
                    tx, ty = t['center']
                    tdist = math.hypot(pcx - tx, pcy - ty)
                    if tdist < 80 and tdist < best_t_dist:
                        best_t_dist = tdist
                        matched_tracker = t

                # Skip OCR computation if tracker is already verified and saved
                if matched_tracker and matched_tracker['saved']:
                    scans_count = len(matched_tracker['plates'])
                    final_plate = matched_tracker['final_plate']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"[SAVED] {final_plate}", (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
                    continue

                plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                plate_text = engine.read_plate(plate_crop)

                if plate_text == "Unknown" or len(plate_text) < 5:
                    continue

                best_match = None
                min_dist = float('inf')
                for veh in cached_vehicles:
                    vx1, vy1, vx2, vy2 = veh['box']
                    vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
                    dist = math.hypot(pcx - vcx, pcy - vcy)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = veh

                v_type = best_match['type'] if best_match else 'car'
                color = best_match['color'] if best_match else "Silver/Gray"

                if matched_tracker:
                    matched_tracker['center'] = (pcx, pcy)
                    matched_tracker['last_seen'] = time.time()
                    matched_tracker['types'].append(v_type)
                    matched_tracker['colors'].append(color)
                    matched_tracker['plates'].append(plate_text)
                else:
                    matched_tracker = {
                        'center': (pcx, pcy),
                        'last_seen': time.time(),
                        'types': [v_type],
                        'colors': [color],
                        'plates': [plate_text],
                        'saved': False
                    }
                    trackers.append(matched_tracker)

                scans_count = len(matched_tracker['plates'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Scanning ({scans_count}/5): {plate_text}", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                if scans_count >= 5 and not matched_tracker['saved']:
                    final_type = Counter(matched_tracker['types']).most_common(1)[0][0]
                    final_color = Counter(matched_tracker['colors']).most_common(1)[0][0]
                    
                    max_len = max(len(p) for p in matched_tracker['plates'])
                    longest_plates = [p for p in matched_tracker['plates'] if len(p) == max_len]
                    final_plate = Counter(longest_plates).most_common(1)[0][0]

                    matched_tracker['final_plate'] = final_plate

                    if not is_duplicate_plate(final_plate, logged_plates):
                        logged_plates.add(final_plate)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), final_type, final_color, final_plate])
                        f.flush()
                        print(f"[VERIFIED MULTI-CAR SCAN] Saved to CSV -> Vehicle: {final_type}, Color: {final_color}, Plate: {final_plate}")
                    
                    matched_tracker['saved'] = True

            current_time = time.time()
            trackers = [t for t in trackers if current_time - t['last_seen'] < 5.0]

            try:
                cv2.imshow("Multi-Car License Plate Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    run_pipeline()
