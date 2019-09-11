import cv2
import scipy.io
import time, os, sys, datetime

def save_new_video_file(cap, output_path):

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%dT%H%M%S')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path + 'video_' + nowDatetime +'.mp4', fourcc, 33.0, (int(cap.get(3)), int(cap.get(4))))
    print('save video : video_' + nowDatetime +'.mp4')

    return out


def video_frame_cropping(input_video_path, output_path, CROP_NUMBER):

    while(True):

        print('Loading WebCam...')
        cap = cv2.VideoCapture(input_video_path)
        cap.set(3,1280)
        cap.set(4,720)
        cap.set(cv2.CAP_PROP_FPS, 33)

        frame_cnt = 0
        save_cnt = 0

        # first save video
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y%m%dT%H%M%S')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path + 'video_' + nowDatetime +'.mp4', fourcc, 33.0, (int(cap.get(3)), int(cap.get(4))))

        while(True):

            ret, image = cap.read()

            if not ret: 
                print("done.")
                break

            frame = image.copy()

            if type(image) is type(None):
                exit()

            print("frame cnt: " + str(frame_cnt))

            out.write(frame)
            frame_cnt += 1
            save_cnt  += 1

            if save_cnt == CROP_NUMBER:
                save_cnt = 0
                out = save_new_video_file(cap, output_path)
                
            if frame_cnt == 0:
                break

            cv2.imshow("test", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

	

if __name__ == '__main__':

    input_video_path = '/home/vtest.mp4'
    output_path = '/home/output/'
    CROP_NUMBER = 100

    video_frame_cropping(input_video_path, output_path, CROP_NUMBER)


