import os
import operator
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19


class ParkTest:

    def __init__(self):
        self.image = cv2.imread('park.jpg')

    def cv_show(self, name, img):
        """"show the image

        :param name and img:
        """
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def select_rgb_white_yellow(self, image):
        """flit the background

        :param image:
        :return masked:
        """
        lower = np.array([120, 120, 120])
        upper = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        return masked

    def convert_gray_scale(self, masked):
        """BGR2GRAY

        :param masked:
        :return gray_image:
        """
        return cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, masked, low_threshold=50, high_threshold=200):
        """using Canny algorithm

        :param masked:
        :param low_threshold:
        :param high_threshold:
        :return edge_image:
        """
        return cv2.Canny(masked, low_threshold, high_threshold)

    def select_region(self, edge_image):
        """save the parking region merely

        using self.filter_region()
        :param edge_image:
        :return roi_image:
        """
        rows, cols = edge_image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]
        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)

        # point_img = edge_image.copy()
        # point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
        # for point in vertices[0]:
        #     cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)

        return self.filter_region(edge_image, vertices)

    def filter_region(self, edge_image, vertices):
        """called by self.select_region()

        :param edge_image:
        :param vertices:
        :return roi_image:
        """
        mask = np.zeros_like(edge_image)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(edge_image, edge_image, mask=mask)

    def hough_lines(self, roi_image):
        """Hough Linear Detection

        :param roi_image:
        :return lines: Lists with linear four coordinates:
        """
        return cv2.HoughLinesP(roi_image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    def identify_blocks(self, roi_image, lines, make_copy=True):
        """select and sort lines in columns

        :param roi_image:
        :param lines: Lists with linear four coordinates:
        :param make_copy: Bool:
        :return new_image:
        :return rect: Dictionary, key: column number, value: rect four coordinates
        """
        global new_image
        if make_copy:
            new_image = np.copy(roi_image)
        cleaned = []  # collect good lines
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:  # exclude lean lines and shout lines as noise
                    cleaned.append((x1, y1, x2, y2))
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))  # sort the good lines according to the positions

        clusters = {}  # collect sorted lines by column, key: column number
        dIndex = 0
        clus_dist = 20  # threshold
        for i in range(len(list1) - 1):
            distance = abs(list1[i + 1][0] - list1[i][0])
            if distance <= clus_dist:  # consider neighbouring lines in the same column
                if dIndex not in clusters.keys():  # start a new column
                    clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])
            else:
                dIndex += 1  # pre-start a new column

        rects = {}  # collect rect four coordinates, key: column number
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 8:  # a column with less than 8 lines is considered an incorrect column
                cleaned = sorted(cleaned, key=lambda tup: tup[1])  # sort lines in the same column by y
                avg_y1 = cleaned[0][1]  # y of the first line in the column
                avg_y2 = cleaned[-1][1]  # y of the last line in the column
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1
        print("Num Parking Lanes: ", len(rects))

        buff = 7  # slight change
        for key in rects:
            # left-up coordinate
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            # right-low coordinate
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        return rects

    def draw_parking(self, image, rects, make_copy=True, color=None, thickness=2, save=True):
        """slight change and return spots Dictionary

        :param image:
        :param rects:
        :param make_copy:
        :param color:
        :param thickness: int.
        :param save: Bool.
        :return new_image: save.
        :return spot_dict: Dictionary, key: car number.
        """
        global spot_dict
        if color is None:
            color = [255, 0, 0]
        global new_image, cur_len
        if make_copy:
            new_image = np.copy(image)
        gap = 15.5  # y-gap between spots is 15.5
        spot_dict = {}  # key: car number, value: four coordinates of a car
        tot_spots = 0  # total number of spots
        # slightly change the rect to make it more accurate
        adj_y1 = {0: -5, 1: -10, 2: 25, 3: -5, 4: 28, 5: 15, 6: 0, 7: -25, 8: -10, 9: -25, 10: 55, 11: -45}
        adj_y2 = {0: -5, 1: -15, 2: 10, 3: -10, 4: 0, 5: 15, 6: 15, 7: -15, 8: 10, 9: 15, 10: -5, 11: 30}

        adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
        adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
        for key in rects:
            tup = rects[key]
            # slight change
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])
            y1 = int(tup[1] + adj_y1[key])
            y2 = int(tup[3] + adj_y2[key])
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw lines in rect
            num_splits = int(abs(y2 - y1) // gap)
            for i in range(0, num_splits + 1):
                y = int(y1 + i * gap)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if 0 < key < len(rects) - 1:
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            if key == 0 or key == (len(rects) - 1):
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # correct the Dictionary
            if key == 0 or key == (len(rects) - 1):  # for two columns of either side who have only one col
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
            else:  # for the other columns who have two cols
                for i in range(0, num_splits + 1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap)
                    x = int((x1 + x2) / 2)
                    spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y + gap)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
            self.cv_show('new_image', new_image)
            self.transform_spot_dict = spot_dict
        return new_image, spot_dict

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        """crop and save the image of every spot

        :param image:
        :param spot_dict:
        :param folder_name:
        :return:
        """
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # crop
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            spot_id = spot_dict[spot]
            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))
            cv2.imwrite(os.path.join(folder_name, filename), spot_img)

    def make_prediction(self, image, model, class_dictionary):
        """preprocess the data

        called by self.predict_on_image()
        :param image:
        :param model:
        :param class_dictionary:
        :return:
        """
        img = image / 255.
        image = np.expand_dims(img, axis=0)  # turn image into 4D tensor
        class_predicted = model.predict(x=image, verbose=1)
        inID = np.argmax(class_predicted[0])  # return the index of the max
        label = class_dictionary[inID]
        return label

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=None, alpha=0.5, save=False):
        """mark the empty spot and save the result

        :param image:
        :param spot_dict:
        :param model:
        :param class_dictionary:
        :param make_copy:
        :param color:
        :param alpha: the weight in addWeighted()
        :param save: whether to save the result.
        :return:
        """
        if color is None:
            color = [0, 255, 0]
        global overlay, new_image
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (32, 32))  # match input size of the model
            label = self.make_prediction(spot_img, model, class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1

        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)
        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        if save:
            filename = 'with_marking.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image


if __name__ == '__main__':

    parktest = ParkTest()

    masked = parktest.select_rgb_white_yellow(parktest.image)

    gray_image = parktest.convert_gray_scale(masked)

    edge_image = parktest.detect_edges(masked, low_threshold=50, high_threshold=200)

    roi_image = parktest.select_region(edge_image)

    parktest.cv_show('roi_image', roi_image)

    lines = parktest.hough_lines(roi_image)

    rects = parktest.identify_blocks(roi_image, lines, make_copy=True)

    new_image, spot_dict = parktest.draw_parking(parktest.image,
                                                 rects,
                                                 make_copy=True,
                                                 color=[255, 0, 0],
                                                 thickness=2,
                                                 save=True)

    parktest.save_images_for_cnn(parktest.image, spot_dict, folder_name='cnn_data')


if __name__ == '__main__':

    files_train = 0
    files_validation = 0
    cwd = os.getcwd()

    folder = 'train_data/train'
    for sub_folder in os.listdir(folder):
        path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
        files_train += len(files)

    folder = 'train_data/test'
    for sub_folder in os.listdir(folder):
        path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
        files_validation += len(files)

    print("train data=", files_train, "teat data=", files_validation)

    # settings for model
    img_width, img_height = 32, 32  # model input shape
    train_data_dir = "train_data/train"
    validation_data_dir = "train_data/test"
    nb_train_samples = files_train  # number of train data
    nb_validation_samples = files_validation  # number of test data
    batch_size = 128
    epochs = 15
    num_classes = 2  # number of labels: two: empty or occupied

    model = VGG19(classes=2, weights=None, input_shape=(img_width, img_height, 3))
    print(model.summary())

    generator = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    train_generator = generator.flow_from_directory(directory=train_data_dir, target_size=(32, 32))
    validation_generator = generator.flow_from_directory(directory=validation_data_dir, target_size=(32, 32))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x=train_generator,
                        verbose=1,
                        epochs=epochs)

    model.save(filepath='models/car2.keras', save_format='keras')

    # model.load_weights(filepath='model/car2.keras')  # skip training and use the saved model

    class_dictionary = {0: 'empty', 1: 'occupied'}
    parktest.predict_on_image(image=parktest.image, model=model, spot_dict=spot_dict, class_dictionary=class_dictionary)






