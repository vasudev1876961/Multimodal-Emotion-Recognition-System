def fuse_emotions(face_emotion, text_emotion):
    """
    Simple fusion logic:
    If both agree → return that
    If different → prioritize face (real-time)
    """

    if face_emotion == text_emotion:
        return face_emotion

    # Priority rule
    return face_emotion