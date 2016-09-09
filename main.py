# -*- coding: cp1251 -*-
import cv2
import sys

# Получение аргументов: путь к изображению для поиска лиц.
imagePath = sys.argv[1]

# Создание и загрузка каскада для распознавания лиц.
# Используем данные для поиска лиц из файла haarcascade_frontalface_default.xml из стандартного набора opencv.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Загрузка изображения для поиска лиц.
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Получение результатов поиска.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print ("Найдено лиц: {0}".format(len(faces)))

# Рисование прямоугольников вокруг найденных лиц и последующий показ полученного изображения.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
