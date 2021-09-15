from typing import Tuple
import torch
import numpy as np
from utils.general import linear_assignment, iou, xxyy_to_xysr, xysr_to_xxyy
from utils.kalmanfilter import KalmanFilter

class ClientPayload:
  """Payload class for cameras and server interaction

  Attributes:
    cameraid: Camera id number
    frame: Image that represent the payload
    reid_feat(list): An array of detected person re-identification features
    reid_xyxy(list): An array of detected person bounding box
  """

  def __init__(self,
               frame: np.array,
               reid_feat: list = None,
               cameraid: int = 0,
               reid_xyxy: list = None) -> None:
    if reid_feat is None:
      reid_feat = []

    if reid_xyxy is None:
      reid_xyxy = []

    self.cameraid = cameraid
    self.frame = frame
    self.reid_feat = reid_feat
    self.reid_xyxy = reid_xyxy


class Track:
  """Class that represent tracked person

  Attributes:
    cameraid: Camera id number
    personid: Person id number
    xyxy: An array of detected person bounding box 
      format: [x1,y1,x2,y2]
  """

  def __init__(self, cameraid: int, personid: int, xyxy: list) -> None:
    self.cameraid = cameraid
    self.personid = personid
    self.xyxy = xyxy

    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    # Transition matrix
    self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])

    # Transformation matrix (Observation to State)
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]])

    self.kf.R[2:, 2:] *= 10.  # observation error covariance
    self.kf.P[4:, 4:] *= 1000.  # initial velocity error covariance
    self.kf.P *= 10.  # initial location error covariance
    self.kf.Q[-1, -1] *= 0.01  # process noise
    self.kf.Q[4:, 4:] *= 0.01  # process noise
    self.kf.x[:4] = xxyy_to_xysr(xyxy)

    self.idle_age = 0

  def update(self, xyxy: list) -> None:
    if len(xyxy):
      
      self.kf.update(xxyy_to_xysr(xyxy))
      self.xyxy = list(xysr_to_xxyy(self.kf.x))
      self.idle_age = 0
    else:
      self.idle_age += 1

  def add_idle_age(self, max_idle_age:int) -> bool:
    self.idle_age += 1
    if self.idle_age > max_idle_age:
      return True
    return False

  def get_iou(self, xyxy: list) -> Tuple:
    if self.kf.x[6] + self.kf.x[2] <= 0:
      self.kf.x[6] *= 0.0
    self.kf.predict()

    return iou(xyxy, list(xysr_to_xxyy(self.kf.x))), self.personid

class TrackDatabase:
  """Class of multiple tracks database

  This class provide management for multiple tracks
  over multiple cameras.

  Attributes:
    num_camera: Number of camera
    max_idle_age: Maximum age for an idle(no input) track
  """

  def __init__(self, num_camera: int, max_idle_age:int) -> None:
    self.num_camera = num_camera
    self._db = []
    for x in range(num_camera):
      self._db.append({})
    self._max_idle_age = max_idle_age

  def _add_all_track_idle_age(self, cameraid:int) -> list:
    expired_tracks = []
    for key, value in self._db[cameraid].items():
      if value.add_idle_age(self._max_idle_age):
        expired_tracks.append(key)

    return expired_tracks

  def _remove_expired_track(self, cameraid:int, expired_tracks:list) -> None:
    for id in expired_tracks:
      self._db[cameraid].pop(id, None)

  def get_iou_matrix(self, cameraid: int, list_xyxy: list, num_person: int) -> np.array:
    result = np.zeros((len(list_xyxy), num_person))

    for i in range(len(list_xyxy)):
      for key, value in self._db[cameraid].items():
        res_iou, personid = value.get_iou(list_xyxy[i])
        result[i, personid] = res_iou

    return result

  def update_track(self, cameraid: int, personid: int, xyxy: list) -> None:
    if personid in self._db[cameraid]:
      self._db[cameraid][personid].update(xyxy)
    else:
      self._db[cameraid][personid] = Track(cameraid, personid, xyxy)

  def update_tracks(self, cameraid:int, xyxy_list: list, detection_map: list) -> dict:
    for detection in detection_map:
      self.update_track(cameraid, detection[1], xyxy_list[detection[0]])

    expired_tracks = self._add_all_track_idle_age(cameraid)
    self._remove_expired_track(cameraid, expired_tracks)
    
    return self._db[cameraid]


class PersonFeat:
  """Class of person's collection of features

  This class provide features store mechanism and some
  re-identification basic operation 

  Attributes:
    personid: Person id number
  """

  def __init__(self, feat: torch.Tensor, personid: int, max_feat: int) -> None:
    self.personid = personid
    self._max_feat = max_feat
    self._feats = [feat]

  def append(self, feat: torch.Tensor) -> None:
    if (len(self._feats) <= self._max_feat):
      self._feats.append(feat)
    else:
      self._feats.pop(0)
      self._feats.append(feat)

  def get_distance(self, feat: torch.Tensor) -> float:
    res = 0
    for item in self._feats:
      res += float(torch.nn.functional.cosine_similarity(item, feat, dim=0))

    return res / len(self._feats)


class PersonDatabase:
  """Class of multiple person features database

  This class provide multi person's features management
  """

  def __init__(self, max_feat: int, distance_threshold: float, 
                iou_weight: float, reid_weigth: float, reid_threshold: float) -> None:
    self._db = []
    self._max_feat = max_feat
    self._distance_threshold = distance_threshold
    self._reid_threshold = reid_threshold
    self._iou_weight = iou_weight
    self._reid_weight = reid_weigth

  def get_num_person(self) -> int:
    return len(self._db)

  def get_track(self, cameraid:int) -> None:
    return self._db[cameraid]

  def add_person(self, feat: torch.Tensor) -> None:
    self._db.append(PersonFeat(feat, len(self._db), self._max_feat))
    return len(self._db) - 1

  def add_feat(self, feat: torch.Tensor, personid: int) -> None:
    self._db[personid].append(feat)

  def _generate_distance_mat(self, list_feat: list) -> np.array:
    result = []
    for feat in list_feat:
      temp = []
      for person in self._db:
        distance = person.get_distance(feat)
        temp.append(distance)
      result.append(temp)
    return np.array(result)

  def feature_matching(self, list_feat: list, iou_mat: np.array) -> list:
    result = []
    if len(list_feat) and len(self._db):
      dis_mat = self._generate_distance_mat(list_feat)
      dis_mat[dis_mat < self._reid_threshold] = 0

      tot_mat = self._reid_weight * dis_mat + self._iou_weight * iou_mat
      tot_mat[tot_mat < self._distance_threshold] = 0
      matched_detection = linear_assignment(-tot_mat)

      unmatched_detection = []
      for feat_id in range(len(list_feat)):
        if feat_id not in matched_detection[:, 0]:
          unmatched_detection.append(feat_id)

      for m in matched_detection:
        if (tot_mat[m[0], m[1]] < self._distance_threshold):
          unmatched_detection.append(m[0])
        else:
          self.add_feat(list_feat[m[0]], m[1])
          result.append([m[0], m[1]])

      for unmatched in unmatched_detection:
        personid = self.add_person(list_feat[unmatched])
        result.append([unmatched, personid])
    
    elif len(list_feat):
      for i in range(len(list_feat)):
        personid = self.add_person(list_feat[i])
        result.append([i, personid])

    return result