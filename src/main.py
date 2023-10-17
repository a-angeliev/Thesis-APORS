import os
import uuid
import cv2
import time

import tensorflow as tf
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


FROM_IMAGE = False

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ///////////////////// nit model loading ///////////////////////////
# Load pipeline config and build a detection model
path1= 'D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_nit'
# path1= 'D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_nit_v1'

configs = config_util.get_configs_from_pipeline_file('D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_nit/pipeline.config')
# configs = config_util.get_configs_from_pipeline_file('D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_nit_v1/pipeline.config')

model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_nit/", 'ckpt-51')).expect_partial()
# ckpt.restore(os.path.join("D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_nit_v1/", 'ckpt-201')).expect_partial()
print("Loaded first model")
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# ///////////////////////////////// head model loading ////////////////////////
# path1_head= 'D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_head_v1'
# path1_head= 'D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_head_v2'
path1_head= 'D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/ssd_mobnet_v2_320_head_v1'

# configs_head = config_util.get_configs_from_pipeline_file('D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_head_v1/pipeline.config')
# configs_head = config_util.get_configs_from_pipeline_file('D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_head_v2/pipeline.config')
configs_head = config_util.get_configs_from_pipeline_file('D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/ssd_mobnet_v2_320_head_v1/pipeline.config')

model_config_head = configs_head['model']
detection_model_head = model_builder.build(model_config=model_config_head, is_training=False)

# Restore checkpoint
ckpt_head = tf.compat.v2.train.Checkpoint(model=detection_model_head)

# ckpt_head.restore(os.path.join("D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/my_ssd_restnet50_v1_fpn_final_head_v1/", 'ckpt-51')).expect_partial()
# ckpt_head.restore(os.path.join("D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/centernet_resnet50_v2_512x512_head_v2/", 'ckpt-187')).expect_partial()
ckpt_head.restore(os.path.join("D:/Coding/Thesis/TensorFlow/workspace/training_demo/models/ssd_mobnet_v2_320_head_v1/", 'ckpt-52')).expect_partial()
print("Loaded second model")

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = label_map_util.create_category_index_from_labelmap("D:/Coding/Thesis/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt",
                                                                    use_display_name=True)


@tf.function
def detect_fn_head(image):
    """Detect objects in image."""

    image_h, shapes_h = detection_model_head.preprocess(image)
    prediction_dict_h = detection_model_head.predict(image_h, shapes_h)
    detections_h = detection_model_head.postprocess(prediction_dict_h, shapes_h)

    return detections_h, prediction_dict_h, tf.reshape(shapes_h, [-1])

category_index_head =  label_map_util.create_category_index_from_labelmap("D:/Coding/Thesis/TensorFlow/workspace/training_demo/annotations-head/annotations/label_map.pbtxt",
                                                                    use_display_name=True)   

print("Define detection functions")



while FROM_IMAGE:

    img_name = input("Image name: ")
    start_a = time.time()
    IMG_PATH = os.path.join("D:\Coding\Thesis\TensorFlow\workspace/training_demo\images/train", f"{img_name}.jpg")
    print("Load image")

    
    start = time.time()
    img = cv2.imread(IMG_PATH)
    image_np = np.array(img)
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    print(f"Detect the nit for time: {time.time() - start}")
    

    boxes = detections['detection_boxes'].numpy()[0]
    max_boxes_to_draw = 1
    score = detections['detection_scores'].numpy()[0]
    min_score_thresh = .9
    coordinates = []
    im_height = image_np.shape[0]
    im_width  = image_np.shape[1]

    if score[0] > min_score_thresh:
        x1 = int(im_width*boxes[0][1])
        x2 = int(im_width*boxes[0][3])
        y1 = int(im_height*boxes[0][0])
        y2 = int(im_height*boxes[0][2])
        crop_img = image_np[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        # crop_img = cv2.resize(crop_img, dsize=620, 620), interpolation=cv2.INTER_CUBIC)


        start = time.time()
        crop_img_np = np.array(crop_img)
        head_input_tensor = tf.convert_to_tensor(np.expand_dims(crop_img_np, 0), dtype=tf.float32)
        detection_head, predictions_dict_head, shapes_head = detect_fn_head(head_input_tensor)
        print(f"Detect head for time: {time.time()-start}")
        

        label_id_offset =  2
        crop_img_with_detections = crop_img.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
            crop_img_with_detections,
            detection_head['detection_boxes'][0].numpy(),
            (detection_head['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detection_head['detection_scores'][0].numpy(),
            category_index_head,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.20,
            agnostic_mode=False)
        cv2.imshow('head', cv2.resize(crop_img_with_detections, (800, 800)))

    print(f"Time for whole detection: {time.time() - start_a}")

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('D:/20230429_153337.mp4')
# cap = cv2.VideoCapture('D:/20230429_153337_1.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while not FROM_IMAGE:
    start_a = time.time()
    start = time.time()
    # Read frame from camera
    ret, img = cap.read()
    height = img.shape[0]
    width  = img.shape[1]
 

    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    print(f"Detect the nit for time: {time.time() - start}")
    
    boxes = detections['detection_boxes'].numpy()[0]
    
    max_boxes_to_draw = 1
    score = detections['detection_scores'].numpy()[0]
    min_score_thresh = .80
    coordinates = []
    im_height = image_np.shape[0]
    im_width  = image_np.shape[1]

    x1 = int(im_width*boxes[0][1])
    x2 = int(im_width*boxes[0][3])
    y1 = int(im_height*boxes[0][0])
    y2 = int(im_height*boxes[0][2])


    img_with_detection = image_np.copy()
    if y1 < 400 and y1>150:
        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_detection,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            line_thickness=1,
            min_score_thresh=.80,
            agnostic_mode=False)
        
        if score[0] > min_score_thresh:

            crop_img = image_np[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)


            start = time.time()
            crop_img_np = np.array(crop_img)
            head_input_tensor = tf.convert_to_tensor(np.expand_dims(crop_img_np, 0), dtype=tf.float32)
            detection_head, predictions_dict_head, shapes_head = detect_fn_head(head_input_tensor)
            print(f"Detect head for time: {time.time()-start}")
            

            label_id_offset = 2
            crop_img_with_detections = crop_img.copy()


            box_color =(255,69,0) 
            viz_utils.visualize_boxes_and_labels_on_image_array(
                crop_img_with_detections,
                detection_head['detection_boxes'][0].numpy(),
                (detection_head['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detection_head['detection_scores'][0].numpy(),
                category_index_head,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.20,
                line_thickness=10,

                agnostic_mode=False)
            cv2.imshow('head', cv2.resize(crop_img_with_detections, (800, 800)))

            # //////////////////////////////////////////////////////////////////////////////
            # Detect head orientaition

            boxes_head = detection_head['detection_boxes'].numpy()[0]
            max_boxes_to_draw_head = 1
            score_head = detection_head['detection_scores'].numpy()[0]
            min_score_thresh_head = .20
            im_height_head = crop_img_np.shape[0]
            im_width_head  = crop_img_np.shape[1]
            
            if score_head[0] > min_score_thresh_head:
                x1_head = int(im_width_head*boxes_head[0][1])
                x2_head = int(im_width_head*boxes_head[0][3])
                y1_head = int(im_height_head*boxes_head[0][0])
                y2_head = int(im_height_head*boxes_head[0][2])
                center = im_height_head/2

                if y1_head<center and y1 < 400 and y1>150:
                    # print(f"////////////////////////////////////  x1 = {x1_head}, x2 = {x2_head}, y1={y1_head}, y2={y2_head}, center={center}")
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        img_with_detection,
                        detections['detection_boxes'][0].numpy(),
                        (detections['detection_classes'][0].numpy() + 1).astype(int),
                        detections['detection_scores'][0].numpy(),
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.80,
                        line_thickness=6,
                        agnostic_mode=False)
                
    cv2.imshow('nit', cv2.resize(img_with_detection, (800, 800)))
    print(f"Time for whole detection: {time.time() - start_a}")

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
 
