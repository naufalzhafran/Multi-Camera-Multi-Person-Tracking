
# Camera client payload class that sent to server
class ClientPayload:
    def __init__(self, frame, reid_feat = [], cameraid = 0, reid_xyxy = []) -> None:
        self.cameraid = cameraid
        self.frame = frame
        self.reid_feat = reid_feat
        self.reid_xyxy = reid_xyxy

# Track class
class Track:
    def __init__(self, cameraid, personid, reid_feat, reid_xyxy):
        self._max_no_input_age = 5 # Constant parameter for every track
        self.cameraid = cameraid
        self.personid = personid
        self.reid_feat = reid_feat
        self.reid_xyxy = reid_xyxy
        self.no_input_age = 0

    def add_no_input_age(self):
        self.no_input_age = self.no_input_age + 1
        
        return self.no_input_age > self._max_no_input_age

    def update(self, reid_feat, reid_xyxy):
        self.reid_feat = reid_feat
        self.reid_xyxy = reid_xyxy
        self.no_input_age = 0