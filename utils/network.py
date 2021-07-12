
# Camera client payload class that sent to server
class ClientPayload:
    def __init__(self, frame, reid_feat = []) -> None:
        self.frame = frame
        self.reid_feat = reid_feat