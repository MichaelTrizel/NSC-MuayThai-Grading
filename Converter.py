import pandas as pd
import mediapipe as mp
import cv2
import os

class Conveter:
    def __init__(self, videos_file_path, output_file_path):
        self.videos_file_path = videos_file_path
        self.output_file_path = output_file_path
    
    def write_video_node_to_csv(self, clip_df, file_name):
        clip_df.to_csv('{}/{}'.format(self.output_file_path, file_name))
    
    def convert_video_to_node(self):
        clip_df = pd.DataFrame()
        columns = ['clip_name', 'threshold']
        for i in range(33):
            columns.append(('{}_x').format(i))
            columns.append(('{}_y').format(i))
            columns.append(('{}_z').format(i))

        curr_data = []
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        mp_pose = mp.solutions.pose

        f = self.videos_file_path
        if os.path.isfile(f) and (f.endswith('.MOV') or f.endswith('.MP4')):
            f = f.replace('\\', '/')
            cap = cv2.VideoCapture(f)
            frameTime = 1
            threshold = 0
            diff = 4
            clip_big_name = 'test'
            clip_sub_name = f.split('/')[-1].split('.')[0]
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                pose_images = []
                while cap.isOpened():
                    threshold += 1
                    showed, frame = cap.read()
                    if showed:
                        frame = cv2.resize(frame, (1920, 1080))
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img.flags.writeable = False
                        results = holistic.process(img)
                        try:
                            landmarks = results.pose_landmarks.landmark
                        except:
                            pass
                        img.flags.writeable = True
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        if threshold % diff == 0:
                            #cv2.imshow('Mediapipe Feed', img)
                            pose_images.append(img)
                            curr_data = []
                            curr_data.append('{}_{}'.format(clip_big_name, clip_sub_name))
                            curr_data.append(threshold)
                            for j in range(0, 33):
                                curr_data.append(landmarks[j].x)
                                curr_data.append(landmarks[j].y)
                                curr_data.append(landmarks[j].z)
                            clip_df = clip_df.append(pd.DataFrame([curr_data], columns=columns), ignore_index=True)
                        if cv2.waitKey(frameTime) & 0xFF == ord('q'):
                            break
                    else:
                        break
                cap.release()
                cv2.destroyAllWindows()

        return clip_df, pose_images