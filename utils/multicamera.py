import torch
import numpy as np


class ClientPayload:
  """Payload class for cameras and server interaction

  Attributes:
    cameraid: Camera id number
    frame: Image that represent the payload
    reid_feat: An array of detected person re-identification features
    reid_xyxy: An array of detected person bounding box
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
    reid_feat: An array of detected person re-identification features
    reid_xyxy: An array of detected person bounding box
  """

  def __init__(self, cameraid: int, personid: int, reid_feat: torch.tensor,
               reid_xyxy: list) -> None:
    self._max_no_input_age = 5  # Constant parameter for every track
    self.cameraid = cameraid
    self.personid = personid
    self.reid_feat = reid_feat
    self.reid_xyxy = reid_xyxy
    self._no_input_age = 0

  def add_no_input_age(self) -> bool:
    self._no_input_age = self._no_input_age + 1

    return self._no_input_age > self._max_no_input_age

  def update(self, reid_feat: torch.tensor, reid_xyxy: list) -> None:
    self.reid_feat = reid_feat
    self.reid_xyxy = reid_xyxy
    self._no_input_age = 0


class PersonFeat:
  """Class of person's collection of features

  This class provide features store mechanism and some
  re-identification basic operation 

  Attributes:
    personid: Person id number
  """

  def __init__(self, feat: torch.tensor, personid: int, max_feat: int) -> None:
    self.personid = personid
    self._max_feat = max_feat
    self._feats = [feat]

  def append(self, feat: torch.tensor) -> None:
    if (len(self._feats) <= self._max_feat):
      self._feats.append(feat)
    else:
      self._feats.pop(0)
      self._feats.append(feat)

  def get_distance(self, feat: torch.tensor) -> float:
    res = 0
    for item in self._feats:
      res += float(torch.nn.functional.cosine_similarity(item, feat, dim=0))

    return res / len(self._feats)


class PersonDatabase:
  """Class of multiple person features database

  This class provide multi person's features management
  """

  def __init__(self, max_feat) -> None:
    self._db = []
    self._max_feat = max_feat

  def add_person(self, feat: torch.tensor) -> None:
    self._db.append(PersonFeat(feat, len(self._db), self._max_feat))

  def add_feat(self, feat: torch.tensor, personid: int) -> None:
    self._db[personid].append(feat)

  def feature_matching(self, list_feat:list) -> list:
    pass