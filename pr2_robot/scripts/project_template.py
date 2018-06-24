#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import os.path


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(ros_msg):
    cloud_filtered = ros_to_pcl(ros_msg)

    cloud_filtered = filter_passthrough_vertical(cloud_filtered)
    cloud_filtered = filter_passthrough_horizontal(cloud_filtered)
    cloud_filtered = filter_voxel_grid(cloud_filtered)
    cloud_filtered = filter_remove_outliers(cloud_filtered)
    pcl_objects = filter_remove_table(cloud_filtered)

    ros_msg_objects = pcl_to_ros(pcl_objects)
    pcl_objects_pub.publish(ros_msg_objects)

    white_cloud = XYZRGB_to_XYZ(pcl_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    cluster_indices = extract_clusters(tree, white_cloud)
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([
                white_cloud[index][0],
                white_cloud[index][1],
                white_cloud[index][2],
                rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    ros_msg_cluster_cloud = pcl_to_ros(cluster_cloud)
    pcl_cluster_cloud_pub.publish(ros_msg_cluster_cloud)

    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = pcl_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def filter_passthrough_vertical(cloud_filtered):
    """Remove points that are higher or below region of interest: parts of the table or robot arms"""

    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)

    axis_min = 0.6
    axis_max = 0.9

    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    return cloud_filtered


def filter_passthrough_horizontal(cloud_filtered):
    """Remove points that are closer or farther the region of interest: parts of the table or robot arms"""

    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)

    axis_min = 0.3
    axis_max = 0.9

    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    return cloud_filtered


def filter_voxel_grid(cloud_filtered):
    """Decrease the number of points by leaving only an average points in each voxel grid cell"""

    vox = cloud_filtered.make_voxel_grid_filter()

    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    return vox.filter()


def filter_remove_outliers(cloud_filtered):
    """Remove alienated points that do not seem to be part of any object"""

    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    outlier_filter.set_mean_k(7)
    outlier_filter.set_std_dev_mul_thresh(0.01)

    cloud_filtered = outlier_filter.filter()
    return cloud_filtered


def filter_remove_table(cloud_filtered):
    """Remove points belonging to a plane which seems to be a table"""

    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    inliers, _ = seg.segment()
    cloud_filtered = cloud_filtered.extract(inliers, negative=True)
    return cloud_filtered


def extract_clusters(tree, white_cloud):
    """Group points into clusters that seem to belong to different objects"""

    ec = white_cloud.make_EuclideanClusterExtraction()

    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(25)
    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract()
    return cluster_indices


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    """Send a move request to PR2 robot"""

    test_scene_num = Int32()
    test_scene_num.data = int(rospy.get_param("scene_number"))

    name_group = { o['name'] : o['group'] for o in rospy.get_param('/object_list') }
    group_pos = { d['group'] : d['position'] for d in rospy.get_param('/dropbox')}

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    dict_list = []
    for object in object_list:
        group = name_group.get(object.label)
        if group is None:
            continue

        arm_name = String()
        if group == "green":
            arm_name.data = "right"
        elif group == "red":
            arm_name.data = "left"
        else:
            continue

        place_pose = Pose()
        group_pos_value = group_pos[group]
        place_pose.position.x = float(group_pos_value[0])
        place_pose.position.y = float(group_pos_value[1])
        place_pose.position.z = float(group_pos_value[2])
        place_pose.orientation.w = 1.0

        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid_np = np.mean(points_arr, axis=0)[:3]

        pick_pose = Pose()
        pick_pose.position.x = float(centroid_np[0])
        pick_pose.position.y = float(centroid_np[1])
        pick_pose.position.z = float(centroid_np[2])
        pick_pose.orientation.w = 1.0

        object_name = String()
        object_name.data = str(object.label)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        yaml_dict = make_yaml_dict(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

    yaml_path = os.path.join(os.path.dirname(__file__), 'output_{0}.yaml'.format(test_scene_num.data))
    send_to_yaml(yaml_path, dict_list)



if __name__ == '__main__':
    rospy.init_node('project_template', anonymous=True)

    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    pcl_cluster_cloud_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    model_path = os.path.join(os.path.dirname(__file__), 'model.sav')
    model = pickle.load(open(model_path, 'rb'))

    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
