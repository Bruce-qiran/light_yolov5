import cv2
import torch
import os

class YOLOv5Detector:
    def __init__(self, model_path, device='cpu'):
        """
        初始化YOLOv5模型
        :param model_path: 本地模型路径，例如 'yolov5s.pt'
        :param device: 运行设备 'cpu' 或 'cuda'
        """                                                        #C:\Users\20601\cra\yolov5\hubconf.py
        self.device = torch.device(device)
        self.model = torch.hub.load('cra/yolov5', 'custom', path=model_path, source='local')  # 本地模型
        self.model.to(self.device)
        self.model.eval()
        print("YOLOv5 模型加载成功！")

    def detect_from_camera(self, camera_index=0):
        """
        从本地摄像头进行检测
        :param camera_index: 本地摄像头的索引，默认0
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("无法打开本地摄像头，请检查设备！")
            return

        print("开始从本地摄像头读取...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break

            # 推理
            results = self.model(frame)

            # 显示带有检测结果的画面
            img_with_boxes = results.render()[0]
            cv2.imshow('YOLOv5 Detection - Local Camera', img_with_boxes)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("摄像头检测结束！")

    def detect_from_image(self, image_path):
        """
        从本地图片进行检测
        :param image_path: 图片的绝对路径
        """
        if not os.path.isfile(image_path):
            print(f"文件路径 {image_path} 不存在，请检查！")
            return

        print(f"开始检测图片：{image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("无法读取图片，请检查路径和文件格式！")
            return

        # 推理
        results = self.model(img)

        # 显示检测结果
        img_with_boxes = results.render()[0]
        cv2.imshow('YOLOv5 Detection - Image', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("图片检测结束！")


# 主程序入口
if __name__ == "__main__":
    model_path = 'yolov5s.pt'  # 模型路径，确保该文件存在本地

    detector = YOLOv5Detector(model_path=model_path, device='cpu')

    # 选择检测模式
    print("请选择检测模式：")
    print("1 - 使用本地摄像头")
    print("2 - 使用本地图片")

    choice = input("请输入选项 (1或2): ")

    if choice == '1':
        detector.detect_from_camera(camera_index=0)  # 0表示本地默认摄像头
    elif choice == '2':
        image_path = r"C:\Users\20601\Desktop\picture2\u=4286534633,1634665346&fm=253&fmt=auto&app=120&f=JPEG.webp"
        
        2  # 替换本地图片路径
        detector.detect_from_image(image_path)
    else:
        print("无效的选项！程序退出。")
