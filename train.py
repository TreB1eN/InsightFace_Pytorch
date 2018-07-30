from config import get_config
from Learner import face_learner

conf = get_config()

learner = face_learner(conf)

learner.train(conf, 8)