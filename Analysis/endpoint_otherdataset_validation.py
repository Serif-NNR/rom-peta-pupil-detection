import glob, cv2
import matplotlib.pyplot as plt

datasets = ["X:\\Pupil Dataset\\Else",
            "X:\\Pupil Dataset\\ExCuSe",
            "X:\\Pupil Dataset\\PupilNet"]


def produce_real_center_points(annot_paths):
    x_constant, y_constant = 1, 1 #320/384, 240/288
    center_dict = dict()
    for path in annot_paths:
        annot_code = path.split('\\')[-1].split('.')[0]
        file = open(path, mode='r', encoding="utf-8")
        context = file.read().split('\n')
        file.close()
        for line in context:
            if line != "":
                part = line.split(" ")
                if part[2] != "X":
                    x = int((float(part[2]) / 2) * x_constant)
                    y = int((288 - (float(part[3]) / 2)) * y_constant)
                    center_dict[annot_code + part[1]] = {"X": x, "Y": y, "RX": part[2], "RY": part[3]}
    return center_dict


def show_center_point_on_image(im_path, cp):
    image = cv2.resize(cv2.imread(im_path), (384, 288), cv2.INTER_AREA)
    image = cv2.circle(image, (cp["X"], cp["Y"]), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("X:\\Pupil Dataset\\CenterPointExample\\"+im_path.split("\\")[-3] + "_" + im_path.split("\\")[-2] + im_path.split("\\")[-1], image)
    #plt.imshow(image)
    #plt.show()




for dataset in datasets:
    print("dataset: ", dataset)
    image_paths = glob.glob(dataset + '\\*\\*.png')
    annot_paths = glob.glob(dataset + '\\*.txt')
    center_dict = produce_real_center_points(annot_paths)
    for im_path in image_paths:
        try:

            im_code, im_no = str(im_path.split("\\")[-2]), str(int(im_path.split("\\")[-1].split(".")[0]))
            #print(im_path, im_no)
            show_center_point_on_image(im_path, center_dict[im_code+im_no])
        except:
            print("I don't analyze this image:", im_path)

