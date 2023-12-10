import math
import cv2
import numpy as np

class CameraCalibration:
    def __init__(self):
        self.object_points = None
        self.camera_matrix = None
        self.rotation_vector = None
        self.translation_vector = None

    def set_parameters(self, object_points, camera_matrix, rotation_vector, translation_vector):
        self.object_points = object_points
        self.camera_matrix = camera_matrix
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector

    def project_points(self):
        
        self.image_points, _ = cv2.projectPoints(self.object_points, self.rotation_vector, self.translation_vector, self.camera_matrix, None)
        return np.squeeze(self.image_points)

    #计算实际检测的矩形和反投影矩形的差值
    def calculate_intersection_over_union(self, real_points2d):
        
        projected_image_points = self.project_points()

        # Calculate the intersection area
        intersection_area = 0

        #计算两个矩形交集矩形的左上角和右下角坐标

        x1_intersection = max(projected_image_points[0][0], real_points2d[0][0])
        y1_intersection = max(projected_image_points[0][1], real_points2d[0][1])

        x2_intersection = min(projected_image_points[2][0], real_points2d[2][0])
        y2_intersection = min(projected_image_points[2][1], real_points2d[2][1])

        # print("交集左上角坐标",x1_intersection,y1_intersection)
        # print("交集右下角坐标",x2_intersection,y2_intersection)

        #计算交集矩形的面积
        intersection_area = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
        # print("交集面积",intersection_area)

        #计算两个矩形并集的面积
        rect1_area = math.fabs(cv2.contourArea(np.array(projected_image_points, dtype=np.float32).astype(int)))
        rect2_area = math.fabs(cv2.contourArea(np.array(real_points2d, dtype=np.float32).astype(int)))
        union_area = rect1_area + rect2_area - intersection_area
        # print("矩形1坐标",projected_image_points)
        # print("矩形1面积",rect1_area)
        # print("矩形2坐标",real_points2d)
        # print("矩形2面积",rect2_area)
        # print("并集面积",union_area)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        # Calculate the intersection over union (IoU) ratio
        iou_ratio = intersection_area / union_area if union_area > 0 else 0.0

        return iou_ratio

# # Example usage:
# # Create a CameraCalibration object and set its parameters
# calibration = CameraCalibration()
# # Set object_points, camera_matrix, rotation_vector, and translation_vector here

# # Calculate the difference between known_image_points and projected_image_points
# known_image_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # Replace with your known image points
# difference = calibration.calculate_difference(known_image_points)
# print("Difference:", difference)




import cv2
import math
import numpy as np

class CameraCalibration:
    def __init__(self):
        self.object_points = None
        self.camera_matrix = None
        self.rotation_vector = None
        self.translation_vector = None

    def set_parameters(self, object_points, camera_matrix, rotation_vector, translation_vector):
        self.object_points = object_points
        self.camera_matrix = camera_matrix
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector

    def project_points(self,yaw):

        # Apply the yaw rotation to the rotation vector
        rotated_rotation_vector = self.rotation_vector.copy()
        rotated_rotation_vector[1] = yaw  # Add the yaw angle to the original rotation vector

        # print("rotated_rotation_vector",rotated_rotation_vector)
        # print("self.translation_vector",self.translation_vector)

        # Project the object points
        image_points, _ = cv2.projectPoints(self.object_points, rotated_rotation_vector, self.translation_vector, self.camera_matrix, None)
        
        return np.squeeze(image_points)

    def calculate_intersection_over_union(self, real_points2d,yaw):
        # Get projected image points for the given yaw
        projected_image_points = self.project_points(yaw)

        # Calculate the intersection area
        intersection_area = 0

        # Calculate coordinates of the intersection rectangle
        x0_intersection = max(projected_image_points[0][0], real_points2d[0][0])
        y0_intersection = max(projected_image_points[0][1], real_points2d[0][1])

        x1_intersection = min(projected_image_points[1][0], real_points2d[1][0])
        y1_intersection = max(projected_image_points[1][1], real_points2d[1][1])

        x2_intersection = min(projected_image_points[2][0], real_points2d[2][0])
        y2_intersection = min(projected_image_points[2][1], real_points2d[2][1])

        x3_intersection = max(projected_image_points[3][0], real_points2d[3][0])
        y3_intersection = min(projected_image_points[3][1], real_points2d[3][1])

        #将x1,x2,x3,x4四个点作为一个矩形，利用counterarea计算矩形的面积
        intersection_area = math.fabs(cv2.contourArea(np.array([[x0_intersection, y0_intersection],
                                                      [x1_intersection, y1_intersection],
                                                      [x2_intersection, y2_intersection],
                                                      [x3_intersection, y3_intersection]]).astype(int)))
        

        # Calculate areas of the two rectangles
        rect1_area = math.fabs(cv2.contourArea(np.array(projected_image_points, dtype=np.float32).astype(int)))
        rect2_area = math.fabs(cv2.contourArea(np.array(real_points2d, dtype=np.float32).astype(int)))

        # Calculate the union area
        union_area = rect1_area + rect2_area - intersection_area

        # Calculate the intersection over union (IoU) ratio
        iou_ratio = intersection_area / union_area if union_area > 0 else 0.0

        return iou_ratio

    def find_maximum_iou_yaw(self, real_points2d, num_samples=100):
        best_yaw = 0
        best_iou = 0

        # Iterate through yaw angles
        for i in range(num_samples):
            yaw = -math.pi/2 + i * (math.pi / num_samples)# Adjust yaw within the specified range
            iou = self.calculate_intersection_over_union(real_points2d, yaw)

            if iou > best_iou:
                best_iou = iou
                best_yaw = yaw

        return best_yaw, best_iou