import socket
import cv2
import numpy as np

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

if __name__ == '__main__':
    # socket init
    server_address = (socket.gethostbyname("drAIver.local"), 10000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server_address)

    key = ''
    while key != ord('q'):

        length_left = recvall(sock, 16)
        stringData_left = recvall(sock, int(length_left))
        data_left = np.fromstring(stringData_left, dtype='uint8')
        decimg_left = cv2.imdecode(data_left, 1)

        cv2.imshow('CLIENT_LEFT', decimg_left)
        key = cv2.waitKey(1) & 0xFF

    cv2.destroyAllWindows()
    sock.close()




