# -*- coding: cp1251 -*-
import cv2
import sys

# ��������� ����������: ���� � ����������� ��� ������ ���.
imagePath = sys.argv[1]

# �������� � �������� ������� ��� ������������� ���.
# ���������� ������ ��� ������ ��� �� ����� haarcascade_frontalface_default.xml �� ������������ ������ opencv.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# �������� ����������� ��� ������ ���.
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ��������� ����������� ������.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print ("������� ���: {0}".format(len(faces)))

# ��������� ��������������� ������ ��������� ��� � ����������� ����� ����������� �����������.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
