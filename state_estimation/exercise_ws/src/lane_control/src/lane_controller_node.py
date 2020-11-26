#!/usr/bin/env python3
import numpy as np
import rospy
import os

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import LaneController
from numpy.linalg import norm


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params['~v_bar'] = DTParam(
            '~v_bar',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=5.0
        )
        self.params['~k_d'] = DTParam(
            '~k_d',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_theta'] = DTParam(
            '~k_theta',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Id'] = DTParam(
            '~k_Id',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Iphi'] = DTParam(
            '~k_Iphi',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~theta_thres'] = rospy.get_param('~theta_thres', None)
        self.params['~d_thres'] = rospy.get_param('~d_thres', None)
        self.params['~d_offset'] = rospy.get_param('~d_offset', None)
        self.params['~integral_bounds'] = rospy.get_param('~integral_bounds', None)
        self.params['~d_resolution'] = rospy.get_param('~d_resolution', None)
        self.params['~phi_resolution'] = rospy.get_param('~phi_resolution', None)
        self.params['~omega_ff'] = rospy.get_param('~omega_ff', None)
        self.params['~verbose'] = rospy.get_param('~verbose', None)
        self.params['~stop_line_slowdown'] = rospy.get_param('~stop_line_slowdown', None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        veh = os.environ['VEHICLE_NAME']
        self.current_pose_source = 'lane_filter'

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbAllPoses,
                                                 "lane_filter",
                                                 queue_size=1)
        self.sub_intersection_navigation_pose = rospy.Subscriber("~intersection_navigation_pose",
                                                                 LanePose,
                                                                 self.cbAllPoses,
                                                                 "intersection_navigation",
                                                                 queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber(f"/{veh}/wheels_driver_node/wheels_cmd",
                                                        WheelsCmdStamped,
                                                        self.cbWheelsCmdExecuted,
                                                        queue_size=1)
        self.sub_stop_line = rospy.Subscriber("~stop_line_reading",
                                              StopLineReading,
                                              self.cbStopLineReading,
                                              queue_size=1)
        self.sub_obstacle_stop_line = rospy.Subscriber("~obstacle_distance_reading",
                                                        StopLineReading,
                                                        self.cbObstacleStopLineReading,
                                                        queue_size=1)

        # self.load_params()
        seg_topic = f"/{veh}/lane_filter_node/seglist_filtered"
        # topic = "/agent/ground_projection_node/lineseglist_out"
        self.sub0 = rospy.Subscriber(seg_topic,
                                     SegmentList,
                                     self.cbSegmentsGround,
                                     queue_size=1)

        self.pub_tj_eq = rospy.Publisher(f"/{veh}/lane_controller_node/trajectory",
                                             Twist2DStamped,
                                             queue_size=1,
                                             dt_topic_type=TopicType.PERCEPTION)
        self.pose_msg = None
        self.mode = "straight"
        self.r_y = 0
        self.state = [np.array([0, 0])] * 5
        self.tj = np.array([0] * 2)
        self.log("Initialized!")

    def load_params(self):
        def _init_dtparam(name):
            str_topics = []
            param_type = ParamType.STRING if name in str_topics else ParamType.FLOAT
            return DTParam(
                f'~{name}',
                param_type=param_type,
                min_value=-100.0,
                max_value=100.0
            )

        param_names = ["L", "th_seg_dist", "th_seg_count", "off_lane", "th_seg_close", "th_seg_far", "wt_slope",
                       "wt_dist", "th_turn_slope", "th_st_slope", "th_lane_slope", "pow_slope", "pow_dist", "v",
                       "wt_omega", "th_omega", "min_slope", "max_L", "exp", "th_turn_slope_low", "th_turn_slope_lowest"]
        self.p = {k: _init_dtparam(k) for k in param_names}

    @staticmethod
    def filter_segments_array(segments_array, distance_threshold):
        try:
            # adjust color
            segments_array[:, 0][segments_array[:, 0] == 2.] = 1.

            # filter length
            segments_array = segments_array[segments_array[:, -1] <= distance_threshold]

            # sort by distance
            segments_array = segments_array[np.argsort(segments_array[:, -1])]
        except IndexError:
            return segments_array
        return segments_array

    @staticmethod
    def parse_segment_list(segments_msg, dist_threshold):
        segments = segments_msg.segments
        data = []
        for s in segments:
            p, col = s.points, s.color
            p1, p2 = (x1, y1), (x2, y2) = (p[0].x, p[0].y), (p[1].x, p[1].y)
            c = np.mean((x1, x2)), np.mean((y1, y2))
            data.append((col, *p1, norm(p1), *p2, norm(p2), *c, norm(c)))
        data = np.array(data)
        data = LaneControllerNode.filter_segments_array(data, dist_threshold)
        return data

    @staticmethod
    def get_equation(segments_array, close_segs, far_segs):
        closest, farthest = segments_array[:close_segs].mean(axis=0), segments_array[-far_segs:].mean(axis=0)
        x1, y1 = closest[-3], closest[-2]
        x2, y2 = farthest[-3], farthest[-2]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        w_eq = np.array((m, b))
        f_pt = np.array((x2, y2))
        return w_eq, f_pt

    @staticmethod
    def get_trajectory(yl, wl, off):
        yellow_line_detected = True if yl is not None else False
        white_line_detected = True if wl is not None else False
        if yellow_line_detected or white_line_detected:
            if not white_line_detected:
                wl = np.array((yl[0] if wl is None else wl[0], yl[1] - 2 * off))
            elif not yellow_line_detected:
                yl = np.array((wl[0] if yl is None else yl[0], wl[1] + 2 * off))
            elif wl[1] is None:
                wl[1] = yl[1]
            elif yl[1] is None:
                yl[1] = wl[1]
        # wl = np.array(
        #     (yl[0] if wl is None or wl[1] is None else wl[0], yl[1] - 2 * off)) if not white_line_detected else wl
        # yl = np.array(
        #     (wl[0] if yl is None or yl[1] is None else yl[0], wl[1] + 2 * off)) if not yellow_line_detected else yl
        # assert yl is not None and wl is not None
        if yl is None and wl is None:
            return np.array([0., 0.])
        return (yl + wl) / 2.

    @staticmethod
    def handle_noise_straight(yl, y_segs, wl, w_segs, off):
        yellow_line_detected = True if yl is not None else False
        white_line_detected = True if wl is not None else False
        right_white = False
        grass_yellow = False

        if white_line_detected:
            # While going straight, if white line is detected on the left side and yellow line is not detected,
            # shift the yellow line two units, and white line 4 units away
            if w_segs[:, -2].mean() > 0:
                # print("white detected on left")
                # yl = np.array((wl[0], wl[1] - 2 * off)) if not yellow_line_detected else yl
                wl = None
                right_white = True

            # If both lines are detected but yellow line is detected on the right side of white line, assume that
            # it is grass and compute trajectory using white line
            elif yellow_line_detected and y_segs[:, -2].mean() < w_segs[:, -2].mean():
                yl = None
                grass_yellow = True
        return yl, wl, grass_yellow, right_white

    def cbSegmentsGround(self, line_segments_msg):
        self.segments_msg = line_segments_msg
        self.load_params()
        data = self.parse_segment_list(line_segments_msg, self.p["th_seg_dist"].value)  # was 0.3
        try:
            white_segments, yellow_segments = data[data[:, 0] == 0.], data[data[:, 0] == 1.]
            n_white_segs, n_yellow_segs = len(white_segments), len(yellow_segments)
        except IndexError:
            n_white_segs, n_yellow_segs = 0, 0
        line_detection_threshold = self.p["th_seg_count"].value
        white_detected, yellow_detected = n_white_segs > line_detection_threshold, n_yellow_segs > line_detection_threshold
        # print(f"Based on n.segs: {white_detected}, {yellow_detected}")
        # If none of the lines are detected, assume that the trajectory has not changed
        gr_f, wr_f, wts_f = False, False, False
        if not white_detected and not yellow_detected:
            w_eq, y_eq, tj_eq, w_f, y_f = self.state

        # If any of the lines is detected, compute the new trajectory
        else:
            w_eq, w_f = LaneControllerNode.get_equation(white_segments, self.p["th_seg_close"].value,
                                                        self.p["th_seg_far"].value) if white_detected else (None, None)
            y_eq, y_f = LaneControllerNode.get_equation(yellow_segments, self.p["th_seg_close"].value,
                                                        self.p["th_seg_far"].value) if yellow_detected else (None, None)
            # print(f"Initial eq using segs: {y_eq}{w_eq}")
            # If taking left turn and only yellow is detected
            prev_m = self.state[2][0]
            if y_eq is not None and w_eq is None and np.abs(prev_m) > self.p["th_turn_slope"].value and prev_m >= 0:
                y_eq = None
            elif w_eq is not None and y_eq is None and np.abs(prev_m) > self.p["th_turn_slope"].value and prev_m < 0:
                w_eq = None
            # w_eq = None
            if w_eq is None and y_eq is None:
                # if w_eq is None and y_eq is None:
                w_eq, y_eq, tj_eq, w_f, y_f = self.state
            else:
                # print(w_eq is None, y_eq is None)
                tj_slope = ((w_eq[0] if w_eq is not None else y_eq[0]) + (
                    y_eq[0] if y_eq is not None else w_eq[0])) / 2.

                wts_f = False
                # Handle any noise
                if np.abs(tj_slope) <= self.p["th_lane_slope"].value:
                    y_eq, w_eq, gr_f, wr_f = self.handle_noise_straight(y_eq, yellow_segments, w_eq, white_segments,
                                                            self.p["off_lane"].value)
                # # While turning right, only use the yellow line
                elif tj_slope < 0 and np.abs(tj_slope) > self.p[
                    "th_lane_slope"].value and yellow_detected and w_eq is not None:
                    w_eq[1] = None
                    wts_f = True
                # w_eq = w_eq * 5. if w_eq is not None else w_eq
                # print(w_eq is None, y_eq is None)
                tj_eq = self.get_trajectory(y_eq, w_eq, self.p["off_lane"].value)

        # y_eq =
        r = self.get_lookahead_point(tj_eq, self.p["L"].value, y_f, n_yellow_segs, w_f, n_white_segs)
        self.state = w_eq, y_eq, tj_eq, w_f, y_f,
        self.tj = tj_eq
        self.r_y = r[1] if r is not None else None
        w_eq = np.array([0, 0]) if w_eq is None else w_eq
        y_eq = np.array([0, 0]) if y_eq is None else y_eq
        print(f"Y:{yellow_detected}{y_eq} W:{white_detected}{w_eq} Gr:{gr_f} Wr:{wr_f} Wts_f:{wts_f}")
        tj_msg = Twist2DStamped()
        tj_msg.v = tj_eq[0]
        tj_msg.omega = tj_eq[1]
        self.publishTj(tj_msg)
        # print(y_eq[0] if y_eq is not None else "None", w_eq[0] if w_eq is not None else "None")
        # if w_eq is not None and y_eq is not None:
        #     print(y_eq, tj_eq, w_eq)
        #     print(self.get_distance_from_line(r, y_eq), self.get_distance_from_line(r, tj_eq), self.get_distance_from_line(r, w_eq))


    @staticmethod
    def get_lookahead_point(tj_eq, l, y_f, n_y, w_f, n_w):
        try:
            r_x = r_x1, r_x2 = np.roots(
                [2 * tj_eq[0] ** 2, 2 * tj_eq[0] * tj_eq[1], tj_eq[1] ** 2 - l ** 2])
            r_y1, r_y2 = tj_eq[0] * r_x + tj_eq[1]
            r1, r2 = np.array([r_x1, r_y1]), np.array([r_x2, r_y2])
            r = None
            if y_f is not None and w_f is not None:
                dir_pt = y_f if n_y > n_w else w_f
            elif y_f is not None and w_f is None:
                dir_pt = y_f
            elif w_f is not None and y_f is None:
                dir_pt = w_f
            if not r:
                r = r1 if norm(dir_pt - r1) < norm(dir_pt - r2) else r2
        except ValueError:
            r = None
        return r

    def cbObstacleStopLineReading(self,msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state  # String of current FSM state

        if self.fsm_state == 'INTERSECTION_CONTROL':
            self.current_pose_source = 'intersection_navigation'
        else:
            self.current_pose_source = 'lane_filter'

        if self.params['~verbose'] == 2:
            self.log("Pose source: %s" % self.current_pose_source)

    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.

        If the source of the message corresponds with the current wanted pose source, it computes a control command.

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """

        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg

            self.pose_msg = input_pose_msg

            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        # print("cb", msg_wheels_cmd.vel_left, msg_wheels_cmd.vel_right)
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishTj(self, tj_msg):
        self.pub_tj_eq.publish(tj_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = (current_s - self.last_s)

        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        else:
               
            # Compute errors
            d_err = pose_msg.d - self.params['~d_offset']
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params['~d_thres']:
                self.log("d_err too large, thresholding it!", 'warn')
                d_err = np.sign(d_err) * self.params['~d_thres']


            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            # print("meh", wheels_cmd_exec)
            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, self.obstacle_stop_line_distance)
                #TODO: This is a temporarily fix to avoid vehicle image detection latency caused unable to stop in time.
                v = v*0.25
                omega = omega*0.25

            else:
                slope = self.tj[0]
                v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance, slope)

            # For feedforward action (i.e. during intersection navigation)
            omega += self.params['~omega_ff']

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header

        # Add commands to car message
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
