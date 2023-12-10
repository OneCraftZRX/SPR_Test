# from position_solver import PositionSolver
# import numpy as np
# import cv2
# po=PositionSolver()
# points_3D = np.array([[-1, 1, 0],
#                       [1, 1, 0],
#                       [-1,-1, 0],
#                       [1,-1, 0]], dtype=np.float64)


# points_2D= np.empty([0,2],dtype=np.float64)
# points_new=np.array([1,2],dtype=np.float64)
# points_2D=np.vstack((points_2D,points_new))
# points_2D=np.vstack((points_2D,points_new))
# # points_2D=np.append(points_2D,[points_new],axis=0)
# print(points_3D)

# fx = 610.32366943
# fy = 610.5026245
# cx = 313.3859558
# cy = 237.2507269
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]], dtype=np.float64)

# distCoeffs =None
# R, t=po.my_pnp(points_3D,points_2D,K,distCoeffs)

# print(t)

# print(t[2])


#####################################################寻找两个圆的交点#############################
import numpy as np
import matplotlib.pyplot as plt

def calculate_circle_intersections(center1, center2, radius):
    d = np.linalg.norm(np.array(center2) - np.array(center1))

    if d > 2 * radius:
        return None  # 两个圆不相交
    else:
        a = (radius ** 2 - radius ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(radius ** 2 - a ** 2)

        x2 = center1[0] + a * (center2[0] - center1[0]) / d
        y2 = center1[1] + a * (center2[1] - center1[1]) / d

        intersection1 = (x2 + h * (center2[1] - center1[1]) / d, y2 - h * (center2[0] - center1[0]) / d)
        intersection2 = (x2 - h * (center2[1] - center1[1]) / d, y2 + h * (center2[0] - center1[0]) / d)

        return intersection1, intersection2


# 初始化画布
radius = 10
fig, ax = plt.subplots()
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

circle1 = plt.Circle((0, 0), radius, fill=False, color='r')
circle2 = plt.Circle((0, 0), radius, fill=False, color='b')
ax.add_patch(circle1)
ax.add_patch(circle2)

intersections, = ax.plot([], [], 'ro')

# 主程序
while True:
    # 生成随机的圆心1和圆心2位置在圆心坐标为（0, 0）的半径为r的圆上
    angle1 = np.random.rand() * 2 * np.pi  # 随机生成角度
    angle2 = np.random.rand() * 2 * np.pi  # 随机生成角度

    center1 = [radius * np.cos(angle1), radius * np.sin(angle1)]
    center2 = [radius * np.cos(angle2), radius * np.sin(angle2)]

    print(center1, center2)
    intersections_data = calculate_circle_intersections(center1, center2, radius)

    if intersections_data is not None:
        x_data, y_data = zip(*intersections_data)
        intersections.set_data(x_data, y_data)

    # 更新圆的位置
    circle1.center = center1
    circle2.center = center2

    # 更新画布
    plt.draw()
    plt.pause(0.1)

