#!/usr/bin/env python3
import math
import os

import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import SegmentList, LanePose, WheelEncoderStamped
from lane_filter import LaneFilterHistogramKF
from sensor_msgs.msg import Image


class LaneFilterNode(DTROS):
    """ Generates an estimate of the lane pose.

    Creates a `lane_filter` to get estimates on `d` and `phi`, the lateral and heading deviation from the center of the lane.
    It gets the segments extracted by the line_detector as input and output the lane pose estimate.


    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use

    Configuration:
        ~filter (:obj:`list`): A list of parameters for the lane pose estimation filter
        ~debug (:obj:`bool`): A parameter to enable/disable the publishing of debug topics and images

    Subscribers:
        ~segment_list (:obj:`SegmentList`): The detected line segments from the line detector
        ~car_cmd (:obj:`Twist2DStamped`): The car commands executed. Used for the predict step of the filter
        ~change_params (:obj:`String`): A topic to temporarily changes filter parameters for a finite time only
        ~switch (:obj:``BoolStamped): A topic to turn on and off the node. WARNING : to be replaced with a service call to the provided mother node switch service
        ~fsm_mode (:obj:`FSMState`): A topic to change the state of the node. WARNING : currently not implemented
        ~(left/right)_wheel_encoder_node/tick (:obj: `WheelEncoderStamped`): Information from the wheel encoders

    Publishers:
        ~lane_pose (:obj:`LanePose`): The computed lane pose estimate

    """

    def __init__(self, node_name):
        super(LaneFilterNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        veh = os.getenv("VEHICLE_NAME")

        self._filter = rospy.get_param('~lane_filter_histogram_kf_configuration', None)
        self._debug = rospy.get_param('~debug', False)
        self._predict_freq = rospy.get_param('~predict_frequency', 30.0)
        print(f"Predict Frequency: {self._predict_freq}")

        # Create the filter
        self.filter = LaneFilterHistogramKF(**self._filter)
        self.t_last_update = rospy.get_time()
        self.last_update_stamp = self.t_last_update

        self.filter.wheel_radius = rospy.get_param(f"/{veh}/kinematics_node/radius")  # 0.0318
        self.filter.wheel_circumference = 2 * math.pi * self.filter.wheel_radius

        # Subscribers
        # segments_topic = f"/{veh}/line_detector_node/segment_list"
        # let_topic = f"/{veh}/left_wheel_encoder_node/tick"
        # ret_topic = f"/{veh}/right_wheel_encoder_node/tick"
        # lane_pose_topic = f"/{veh}/lane_filter_node/lane_pose"
        # ml_topic = f"/{veh}/lane_filter_node/measurement_likelihood_img"
        # filt_seg_topic = f"/{veh}/lane_filter_node/seglist_filtered"

        segments_topic = "~segment_list"
        let_topic = "~left_wheel_encoder_node/tick"
        ret_topic = "~right_wheel_encoder_node/tick"
        lane_pose_topic = "~lane_pose"
        ml_topic = "~measurement_likelihood_img"
        filt_seg_topic = "~seglist_filtered"

        self.sub_segment_list = rospy.Subscriber(segments_topic,
                                                 SegmentList,
                                                 self.cbProcessSegments,
                                                 queue_size=1)

        self.sub_encoder_left = rospy.Subscriber(let_topic,
                                                 WheelEncoderStamped,
                                                 self.cbProcessLeftEncoder,
                                                 queue_size=1)



        self.sub_encoder_right = rospy.Subscriber(ret_topic,
                                                  WheelEncoderStamped,
                                                  self.cbProcessRightEncoder,
                                                  queue_size=1)

        # Publishers
        self.pub_lane_pose = rospy.Publisher(lane_pose_topic,
                                             LanePose,
                                             queue_size=1,
                                             dt_topic_type=TopicType.PERCEPTION)

        self.pub_ml_img = rospy.Publisher(ml_topic,
                                          Image,
                                          queue_size=1,
                                          dt_topic_type=TopicType.DEBUG)

        self.pub_seglist_filtered = rospy.Publisher(filt_seg_topic,
                                                    SegmentList,
                                                    queue_size=1,
                                                    dt_topic_type=TopicType.DEBUG)

        self.right_encoder_ticks = 0
        self.left_encoder_ticks = 0

        # Go to max 18-19 on full speed at v_max 1
        self.right_encoder_ticks_delta = 0
        self.left_encoder_ticks_delta = 0
        # Set up a timer for prediction (if we got encoder data) since that data can come very quickly
        rospy.Timer(rospy.Duration(1 / self._predict_freq), self.cbPredict)

        self.bridge = CvBridge()

    # These set delta
    def cbProcessLeftEncoder(self, left_encoder_msg):
        if not self.filter.initialized:
            self.filter.encoder_resolution = left_encoder_msg.resolution
            self.filter.initialized = True
        self.left_encoder_ticks_delta = left_encoder_msg.data - self.left_encoder_ticks

    def cbProcessRightEncoder(self, right_encoder_msg):
        if not self.filter.initialized:
            self.filter.encoder_resolution = right_encoder_msg.resolution
            self.filter.initialized = True
        self.right_encoder_ticks_delta = right_encoder_msg.data - self.right_encoder_ticks

    # These are called at certain intervals. Updates belief given current delta. Records total ticks. Calls publish estimate.
    def cbPredict(self, event):
        current_time = rospy.get_time()
        dt = current_time - self.t_last_update
        self.t_last_update = current_time

        # first let's check if we moved at all, if not abort
        if self.right_encoder_ticks_delta == 0 and self.left_encoder_ticks_delta == 0:
            return

        self.filter.predict(dt, self.left_encoder_ticks_delta, self.right_encoder_ticks_delta)
        self.left_encoder_ticks += self.left_encoder_ticks_delta
        self.right_encoder_ticks += self.right_encoder_ticks_delta
        # print(
        #     f"LDT: {self.left_encoder_ticks_delta} LTD: {(self.left_encoder_ticks_delta * self.filter.wheel_circumference) / 360.:.3f} "
        #     f"RDT: {self.right_encoder_ticks_delta} RTD: {(self.right_encoder_ticks_delta * self.filter.wheel_circumference) / 360.:.3f}")
        self.left_encoder_ticks_delta = 0
        self.right_encoder_ticks_delta = 0

        self.publishEstimate()

    def cbProcessSegments(self, segment_list_msg):
        """Callback to process the segments

        Args:
            segment_list_msg (:obj:`SegmentList`): message containing list of processed segments

        """

        self.last_update_stamp = segment_list_msg.header.stamp

        # Get actual timestamp for latency measurement
        timestamp_before_processing = rospy.Time.now()

        # Step 2: update
        self.filter.update(segment_list_msg.segments)

        self.publishEstimate(segment_list_msg)

    def publishEstimate(self, segment_list_msg=None):
        belief = self.filter.getEstimate()
        pred = self.filter.process_out
        meas = self.filter.measurement_out
        ekf = self.filter.ekf_out
        # print(f"PRO:{pred[0]:.2f} {pred[1]:.2f} MEA:{meas[0]:.2f} {meas[1]:.2f}\t EKF:{ekf[0]:.2f} {ekf[1]:.2f}")

        # build lane pose message to send
        lanePose = LanePose()
        lanePose.header.stamp = self.last_update_stamp
        lanePose.d = belief['mean'][0]
        lanePose.phi = belief['mean'][1]
        lanePose.d_phi_covariance = [belief['covariance'][0][0],
                                     belief['covariance'][0][1],
                                     belief['covariance'][1][0],
                                     belief['covariance'][1][1]]
        lanePose.in_lane = True
        lanePose.status = lanePose.NORMAL

        self.pub_lane_pose.publish(lanePose)
        if segment_list_msg is not None:
            self.debugOutput(segment_list_msg, lanePose.d, lanePose.phi)

    def debugOutput(self, segment_list_msg, d_max, phi_max):
        """Creates and publishes debug messages

        Args:
            segment_list_msg (:obj:`SegmentList`): message containing list of filtered segments
            d_max (:obj:`float`): best estimate for d
            phi_max (:obj:``float): best estimate for phi

        """
        if self._debug:

            # Get the segments that agree with the best estimate and publish them
            inlier_segments = self.filter.get_inlier_segments(segment_list_msg.segments,
                                                              d_max,
                                                              phi_max)
            inlier_segments_msg = SegmentList()
            inlier_segments_msg.header = segment_list_msg.header
            inlier_segments_msg.segments = inlier_segments

            self.pub_seglist_filtered.publish(inlier_segments_msg)

            # Create belief image and publish it
            ml = self.filter.generate_measurement_likelihood(segment_list_msg.segments)
            if ml is not None:
                ml_img = self.bridge.cv2_to_imgmsg(
                    np.array(255 * ml).astype("uint8"), "mono8")
                ml_img.header.stamp = segment_list_msg.header.stamp
                self.pub_ml_img.publish(ml_img)

    def cbMode(self, msg):
        return  # TODO adjust self.active

    def updateVelocity(self, twist_msg):
        self.currentVelocity = twist_msg

    def loginfo(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    lane_filter_node = LaneFilterNode(node_name="lane_filter_node")
    rospy.spin()
