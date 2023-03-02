import cv2
import os
import numpy
import datetime
import pandas as pd
import pyttsx3
import geocoder

from listaPermitidos import flabianos

# Inicializar la librería pyttsx3
engine = pyttsx3.init()

# Obtener una lista de usuarios permitidos
flabs = flabianos()

print('Tomando asistencia...')

dir_faces = 'ucv/img'
size = 4

g = geocoder.ip('me')
lat,lng = g.latlng
print('Latitud', lat)
print('Longuitud:', lng)

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(dir_faces):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dir_faces, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(im_width, im_height) = (112, 92)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Crear un DataFrame vacío con las columnas requeridas
df = pd.DataFrame(columns=['User', 'Timestamp', 'Status'])

while True:
    rval, frame = cap.read()
    frame = cv2.flip(frame, 1, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    faces = face_cascade.detectMultiScale(mini)

    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        prediction = model.predict(face_resize)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cara = '%s' % (names[prediction[0]])

        if prediction[1] < 100:
            cv2.putText(frame, '%s - %.0f' % (cara, prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # Add the detected username and current date and time to the DataFrame
            now = datetime.datetime.now()
            
            # Check if the person is late or on time
            if now.hour >= 8:
                status = 'Tarde'
                color = (0, 0, 255)  # Rojo
            else:
                status = 'Temprano'
                color = (0, 255, 0)  # Verde

                df = pd.concat([df, pd.DataFrame({'User': [cara], 'Timestamp': [now], 'Status': [status]})], ignore_index=True)

                # Convertir la fecha a un formato legible
                fecha = now.strftime("%d de %B del %Y a las %H:%M:%S")

                #   Definir el mensaje a hablar
                mensaje = f"Bienvenido a la UCV, {cara}. La fecha y hora actual es {fecha} Usted a llegado {status}"

                # Hacer que el sistema hable el mensaje
                engine.say(mensaje)
                engine.runAndWait()

                # Print a message in the console
                print("Se ha reconocido a %s a las %s" % (cara, now.strftime("%Y-%m-%d %H:%M:%S")))

        elif prediction[1] > 101 and prediction[1] < 500:
            cv2.putText(frame, 'Desconocido', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('OpenCV Face Recognition', frame)

    key = cv2.waitKey(10)
    if key == 27:
        # Save the DataFrame to an Excel file before exiting the program
        df.to_excel('ucv.xlsx', index=False)
        cv2.destroyAllWindows()
        break

