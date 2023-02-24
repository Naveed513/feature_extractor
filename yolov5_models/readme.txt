This section pertains to the storage location of YOLOv5 files that are utilized for processing. The steps to train a YOLOv5 model are listed below:

Step 1: Preparing Annotations
        Annotations are necessary for training the YOLOv5 model. You can use tools like Roboflow to prepare annotations. The following code installs Roboflow, retrieves the annotations, and downloads the dataset:

        !pip install roboflow

        from roboflow import Roboflow
        rf = Roboflow(api_key="rLrbqUtzNTvid80vdkRU")
        project = rf.workspace("naveed-sk").project("feature-extractor")
        dataset = project.version(3).download("yolov5")

Step 2: Model Training
        To train the model, you can use Google Colab. The following steps are involved:

        a) Clone YOLOv5 to a local path, install the necessary requirements and check the installation status:

        !git clone https://github.com/ultralytics/yolov5  # clone
        %cd yolov5
        %pip install -qr requirements.txt  # install
        import torch
        import utils
        display = utils.notebook_init()  # checks

        b) Load the trained dataset from the previous step using:

        !pip install roboflow
        from roboflow import Roboflow
        rf = Roboflow(api_key="rLrbqUtzNTvid80vdkRU")
        project = rf.workspace("naveed-sk").project("feature-extractor")
        dataset = project.version(3).download("yolov5")

        c) Modify the paths of train, test, and validation in the data.yaml file located in the annotations.

        d) Train the model using the following command:

        #!python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
        # change the coco128.yaml with the data.yaml path in annotations and epochs can be increased as per requirement i changed to 250
        !python train.py --img 640 --batch 16 --epochs 250 --data /content/feature-extractor/data.yaml --weights yolov5s.pt --cache
        
Step 3: Checking Model Performance with Test Cases
        To evaluate the performance of the trained YOLOv5 model with a test case, you can utilize the following code:


        # YOLOv5 PyTorch HUB Inference (DetectionModels only)
        # import torch
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)  # yolov5n - yolov5x6 or custom

        model = torch.hub.load('ultralytics/yolov5', 'custom', '/content/feature-extractor/runs/exp/weights/best.pt', force_reload=True)
        im = 'https://ultralytics.com/images/zidane.jpg'  # file, Path, PIL.Image, OpenCV, nparray, list
        results = model(im)  # inference
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

These steps provide a comprehensive guide to train and test a YOLOv5 model.
