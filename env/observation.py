import numpy as np
import cv2

class ObservationPipeline:

    def process_image(self, image):

        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        img = cv2.resize(img, (84,84))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32) / 255.0
        return img[None,:,:]

    def vector_state(self, vehicle):

        vel = vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2) ** 0.5

        return np.array([speed], dtype=np.float32)